from hmb.load import SequencingLoader
from hmb.management.import_base import AbstractImportCommand


class Command(AbstractImportCommand):
    loader_class = SequencingLoader
    help = 'Import data from meta file'
