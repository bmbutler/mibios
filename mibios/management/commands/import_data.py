from mibios.dataset import registry
from mibios.load import GeneralLoader
from mibios.management.import_base import AbstractImportCommand


class Command(AbstractImportCommand):
    loader_class = GeneralLoader
    help = 'Import data for pre-defined dataset/table formats'

    def add_arguments(self, argp):
        argp.add_argument(
            '-d', '--data-set-name', '--table', '--model',
            required=True,
            help='Name of table, model, or data set for which to import data. '
                 'Available table names are: '
                 + ', '.join(registry.get_names()),
        )

    def load_file_kwargs(self, **options):
        return dict(dataset=options['data_set_name'])
