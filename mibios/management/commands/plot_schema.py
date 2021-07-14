from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles.finders import find as find_static
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = ('convenience command to plot the schema, a wrapper around '
            'graph_models from the django_extensions app')

    def handle(self, *args, **options):
        if 'django_extensions' not in settings.INSTALLED_APPS:
            raise CommandError('this command depends on having the '
                               'django_extensions app installed')

        static_root = find_static('mibios')
        if static_root is None:
            raise CommandError('failed to obtain static files root for '
                               'mibios app')

        schema_path = Path(static_root) / settings.SCHEMA_PLOT_PATH
        if not schema_path.parent.is_dir():
            try:
                schema_path.parent.mkdir()
            except Exception as e:
                raise CommandError() from e

        call_command(
            'graph_models',
            *settings.SCHEMA_PLOT_APPS,
            group_models=True,
            output=str(schema_path),
            exclude_models=settings.SCHEMA_PLOT_EXCLUDE,
            verbose_names=True,
        )
        self.stdout.write(f'image saved to: {schema_path}')

        schema_path = Path(static_root) / settings.SCHEMA_PLOT_SIMPLE_PATH
        excl = [settings.SCHEMA_PLOT_EXCLUDE]
        excl += [
            'Model',
            'ChangeRecord'
            'ContigAbundance',
            'ContigCluster',
            'ContigClusterMembership',
            'ContigClusterLevel',
            'GeneCluster',
            'GeneClusterMembership',
            'GeneClusterLevel',
        ]
        call_command(
            'graph_models',
            *settings.SCHEMA_PLOT_APPS,
            group_models=True,
            output=str(schema_path),
            exclude_models=excl,
            verbose_names=True,
        )
        self.stdout.write(f'image saved to: {schema_path}')
