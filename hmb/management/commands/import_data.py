from hmb.dataset import DATASET
from hmb.load import GeneralLoader
from hmb.management.import_base import AbstractImportCommand


class Command(AbstractImportCommand):
    loader_class = GeneralLoader
    help = 'Import data for pre-defined dataset/table formats'

    def add_arguments(self, argp):
        argp.add_argument(
            '-d', '--data-set-name', '--table', '--model',
            required=True,
            help='Name of table, model, or data set for which to import data. '
                 'Available table names are: ' + ', '.join(DATASET.keys()),
        )

    def load_file_kwargs(self, **options):
        return dict(dataset=options['data_set_name'])
