from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = ('Print table of current users that can be edited and then used as '
            'input gor the update_users command')

    def print_row(self, *row):
        self.stdout.write(' '.join([str(i) for i in row if i]) + '\n')

    def handle(self, *args, **options):
        for i in User.objects.filter(is_active=True):
            perms = i.user_permissions.all()
            for j in perms:
                # using single change permission as proxy for allowing any
                # changes
                if j.codename.startswith('change_'):
                    can_edit = True
                    break
            else:
                can_edit = False

            if i.email == i.username + '@umich.edu':
                email = None
            else:
                email = i.email

            row = [
                i.username,
                email, 
                'editor' if can_edit else '',
                'superuser' if i.is_superuser else '',
            ]
            self.print_row(*row)
