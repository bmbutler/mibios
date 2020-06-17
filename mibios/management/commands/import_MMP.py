from mibios.load import MMPManifestLoader
from mibios.management.import_base import AbstractImportCommand


class Command(AbstractImportCommand):
    loader_class = MMPManifestLoader
    help = 'Import data from MMP_manifest file'

    DEFAULT_SEPARATOR = ','

    def add_arguments(self, argp):
        argp.add_argument(
            '-w', '--warn-on-error',
            action='store_true',
            help='Warn about and skip lines in the input data that can not be '
                 'processed because the input does not conform to expectations'
                 '; the default is to abort on any error and not to apply any '
                 'changed to the database.'
        )
        argp.add_argument(
            '--strict-sample-id',
            action='store_true',
            help='Raise error on non-conforming sample IDs.  By default, these'
                 ' are ignored in the sense that a sequencing record is still '
                 'created but without a corresponding fecal sample.  This is '
                 'because we don\'t consider the MMP_manifest file '
                 'as authorative with respect to the sample ids.  Conforming '
                 'IDs follow the usual Unnn_n scheme.'
        )

    def load_file_kwargs(self, **options):
        return dict(
            warn_on_error=options['warn_on_error'],
            strict_sample_id=options['strict_sample_id'],
        )
