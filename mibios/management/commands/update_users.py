from django.db import transaction
from django.contrib.auth.models import Permission, User
from django.core.management.base import BaseCommand


EDITOR = 'editor'
SUPERUSER = 'superuser'


class Command(BaseCommand):
    help = 'Update user data base from file'

    def add_arguments(self, argp):
        argp.add_argument(
            'usertable',
            help='Input file, composed of lines that list (space-separated)'
                 'the uniqname, email, and, as required, the keywords editor '
                 'and/or superuser',
        )
        argp.add_argument(
            '-k', '--keep',
            action='store_true',
            help='Keep existing users listed not in input table.  The default '
                 'is to delete them.',
        )

    @transaction.atomic
    def handle(self, *args, **options):
        with open(options['usertable']) as f:
            view_perms = Permission.objects.filter(
                codename__startswith='view_',
                content_type__app_label='mibios',
            )

            edit_perms = Permission.objects.filter(
                content_type__app_label='mibios',
            ).exclude(
                codename__startswith='view_',
            )

            present_users = []
            for line in f:
                row = line.strip().split()
                if not row or row[0].startswith('#'):
                    continue

                uniquename = row[0]
                is_editor = False
                is_superuser = False
                email = None

                if len(row) >= 2:
                    if '@' in row[1]:
                        email = row[1]
                    if EDITOR in row:
                        is_editor = True
                    if SUPERUSER in row:
                        is_superuser = True

                user, new = User.objects.get_or_create(username=uniquename)
                if new:
                    self.stdout.write('new user: {}'.format(user))
                present_users.append(user.username)

                if new and not email:
                    email = uniquename + '@umich.edu'

                if email:
                    user.email = email

                user.is_superuser = is_superuser
                user.is_staff = True  # all can access the admin interface
                user.save()
                user.user_permissions.clear()
                user.user_permissions.add(*view_perms)

                if is_editor:
                    user.user_permissions.add(*edit_perms)


            if not options['keep']:
                qs = User.objects.exclude(username__in=present_users)
                if qs.exists():
                    self.stdout.write('Removing {} users:')
                    for i in qs:
                        self.stdout.write(i.username)
                    qs.delete()

        self.stdout.write('done')
