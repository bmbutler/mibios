import argparse

from django.core.management.base import BaseCommand, CommandError

from hmb.load import SequencingLoader, UserDataError


class AbstractImportCommand(BaseCommand):
    # Implementers must loader class, which must be
    # an implementation of hmb.load.AbstractLoader
    loader_class = None

    DEFAULT_SEPARATOR = '\t'

    def load_file_kwargs(self, **options):
        """
        Override to pass additional kwargs to load_file()

        Return value must be a dict and is directly passed to the loders
        load_file method
        """
        return {}

    def create_parser(self, prog_name, subcommand):
        """
        Add common options

        calls super and adds options
        """
        parser = super().create_parser(prog_name, subcommand)
        parser.add_argument(
            'file',
            type=argparse.FileType(),
            help='Inputfile, tab-separated with columns: '
                 '' + ', '.join([i[0] for i in SequencingLoader.COLS]),
        )
        parser.add_argument(
            '-s', '--sep',
            default=self.DEFAULT_SEPARATOR,
            help='Column separator, default is <tab>',
        )
        return parser

    def handle(self, *args, **options):
        self.stdout.write('Loading {} ...'.format(options['file'].name))
        try:
            count, new, changed = self.loader_class.load_file(
                options['file'],
                sep=options['sep'],
                **self.load_file_kwargs(**options),
            )
        except UserDataError as e:
            raise CommandError(e)
        except Exception as e:
            raise CommandError('Failed importing data') from e

        self.stdout.write('{} rows processed'.format(count))
        if new:
            self.stdout.write('Imported:\n' + '\n'.join([
                '  {}: {}'.format(k, v)
                for k, v
                in new.items()
            ]))
        else:
            self.stdout.write('No new records\n')

        if changed:
            self.stdout.write('Modified:\n' + '\n'.join([
                '  {}: {}'.format(k, v)
                for k, v
                in changed.items()
            ]))
        else:
            self.stdout.write('No records changed\n')

        self.stdout.write('All done.')
