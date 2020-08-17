from django.db import transaction
from django.contrib.auth.models import Group, Permission, User
from django.core.management.base import BaseCommand

from mibios.dataset import registry


EDITOR = 'editor'
SUPERUSER = 'superuser'


class Command(BaseCommand):
    help = 'Update user data base from file'

    def add_arguments(self, argp):
        argp.add_argument(
            'usertable',
            help='Input file, composed of lines that list (space-separated)'
                 'the uniqname, email, and, as required, the keywords "'
                 + EDITOR + '" and/or "' + SUPERUSER + '"',
        )
        argp.add_argument(
            '-k', '--keep',
            action='store_true',
            help='Keep users which are not listed in input table active.  The '
                 'default is to de-activate such users.',
        )

    @transaction.atomic
    def handle(self, *args, **options):

        # update groups (all permissions reset)
        apps = ['mibios']
        apps += registry.apps.keys()
        view_perms = Permission.objects.filter(
            codename__startswith='view_',
            content_type__app_label__in=apps,
        )

        edit_perms = Permission.objects.filter(
            content_type__app_label__in=apps,
        ).exclude(
            codename__startswith='view_',
        )

        viewers, new = Group.objects.get_or_create(name='viewers')
        if new:
            self.stdout.write('new group: {}'.format(viewers))
        viewers.permissions.set(view_perms)

        curators, new = Group.objects.get_or_create(name='curators')
        if new:
            self.stdout.write('new group: {}'.format(curators))
        curators.permissions.set(view_perms)
        curators.permissions.add(*edit_perms)

        # update users
        with open(options['usertable']) as f:
            present_users = []
            for line in f:
                row = line.strip().split()
                if not row or row[0].startswith('#'):
                    continue

                uniquename = row[0]
                group = 'viewers'
                is_superuser = False
                email = None

                if len(row) >= 2:
                    if '@' in row[1]:
                        email = row[1]
                    if EDITOR in row:
                        group = 'curators'
                    if SUPERUSER in row:
                        is_superuser = True

                # processing user
                user, new = User.objects.get_or_create(username=uniquename)
                if new:
                    self.stdout.write('new user: {}'.format(user))
                present_users.append(user.username)

                if new and not email:
                    email = uniquename + '@umich.edu'

                if email:
                    user.email = email

                user.is_superuser = is_superuser
                user.is_staff = True  # basic access to admin interface for all
                user.is_active = True
                user.save()

                user.user_permissions.clear()
                user.groups.clear()

                # processing group membership
                group_qs = Group.objects.filter(name=group)
                group = user.groups.set(group_qs)

            if not options['keep']:
                qs = User.objects.exclude(username__in=present_users)
                qs = qs.filter(is_active=True)
                for i in qs:
                    i.is_active = False
                    i.save()
                    self.stdout.write(
                        'user de-activated: {}'.format(i.username)
                    )

        self.stdout.write('done')
