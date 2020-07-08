import argparse

from django.core.management.base import BaseCommand, CommandError

from mibios.load import UserDataError
from mibios.utils import getLogger


class AbstractImportCommand(BaseCommand):
    # Implementers must loader class, which must be
    # an implementation of mibios.load.AbstractLoader
    loader_class = None

    DEFAULT_SEPARATOR = '\t'

    def load_file_kwargs(self, **options):
        """
        Override to pass additional kwargs to load_file()

        Return value must be a dict and is directly passed to the loaders
        load_file method
        """
        return {}

    def create_parser(self, prog_name, subcommand):
        """
        Add common options

        calls super and adds options
        """
        more_help_file = ''
        if hasattr(self.loader_class, 'COLS'):
            more_help_file += ', recognized columns are: '
            more_help_file += ', '.join([i[0] for i in self.loader_class.COLS])

        parser = super().create_parser(prog_name, subcommand)
        parser.add_argument(
            'file',
            type=argparse.FileType(),
            help='Inputfile, tab-separated' + more_help_file,
            )
        parser.add_argument(
            '-n', '--dry-run',
            action='store_true',
            help='Go through all the action but do not commit the data to the '
                 'database',
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite non-blank/non-null values in database with '
                 'corresponding data from the imported table.  By default '
                 'data is added to blank fields only and the existing data is '
                 'deemed authorative and a warning will be issued when values '
                 'in the table differ from those already stored in the '
                 'database.',
        )
        parser.add_argument(
            '-s', '--sep',
            default=self.DEFAULT_SEPARATOR,
            help='Column separator, default is <tab>',
        )
        parser.add_argument(
            '--verbose-changes',
            action='store_true',
            help='List each change of existing values, the default is to just '
                 'give a summary',
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Turn on debugging output'
        )
        return parser

    def handle(self, *args, **options):
        logger = getLogger('mibios')
        if options['debug']:
            logger.setLevel('DEBUG')
        else:
            logger.setLevel('INFO')

        self.stdout.write('Loading {} ...'.format(options['file'].name))
        try:
            stats = self.loader_class.load_file(
                options['file'],
                sep=options['sep'],
                can_overwrite=options['overwrite'],
                dry_run=options['dry_run'],
                **self.load_file_kwargs(**options),
            )
        except UserDataError as e:
            raise CommandError(e)
        except Exception as e:
            raise CommandError('Failed importing data') from e

        kwargs = dict(**options)
        kwargs.update(**stats)
        self.stdout.write(self.format_import_stats(**kwargs))
        self.stdout.write(' All done.')

    @classmethod
    def format_import_stats(cls, count=0, new={}, added={}, changed={},
                            ignored=[], **options):
        out = ''
        if options.get('dry_run', False):
            out += ' (dry run)'
        out += ' {} rows processed\n'.format(count)
        if ignored:
            out += ' {} column(s) not processd: '.format(len(ignored))
            out += ', '.join(ignored) + '\n'
        if new:
            out += ' Imported:\n' + '\n'.join([
                '  {}: {}'.format(k, v)
                for k, v
                in new.items()
            ]) + '\n'
        else:
            out += ' No new records\n'

        if added:
            out += ' Records with blank fields filled:\n' + '\n'.join([
                '  {}: {}'.format(k, v)
                for k, v
                in added.items()
            ]) + '\n'

        if changed:
            if options.get('overwrite'):
                msg = ' Modified:\n'
            else:
                msg = (' Modifications below not applied (no-overwrite/'
                       'append-only option in use)\n')
            out += msg + '\n'.join([
                '  {}: {}'.format(k, len(v))
                for k, v
                in changed.items()
            ]) + '\n'
            if options.get('verbose_changes'):
                for m, i in changed.items():
                    for obj, change_list in i:
                        row = []
                        for field, old, new in change_list:
                            row.append('{}: {} -> {}'.format(field, old, new))
                        out += '   {} {}: {}\n'.format(m, obj, ' | '.join(row))

        else:
            out += ' No records changed\n'
        return out
