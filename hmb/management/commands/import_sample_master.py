from hmb.load import SampleMasterLoader
from hmb.management.import_base import AbstractImportCommand


class Command(AbstractImportCommand):
    loader_class = SampleMasterLoader
    help = 'Import data from "sample master file"'
