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

        Return value must be a dict and is directly passed to the loaders
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
        return parser

    def handle(self, *args, **options):
        self.stdout.write('Loading {} ...'.format(options['file'].name))
        try:
            count, new, added, changed = self.loader_class.load_file(
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

        self.stdout.write(' {} rows processed'.format(count))
        if new:
            self.stdout.write(' Imported:\n' + '\n'.join([
                '  {}: {}'.format(k, v)
                for k, v
                in new.items()
            ]))
        else:
            self.stdout.write(' No new records\n')

        if added:
            self.stdout.write(
                ' Records with blank fields filled:\n' + '\n'.join([
                    '  {}: {}'.format(k, v)
                    for k, v
                    in added.items()
                ])
            )

        if changed:
            if options['overwrite']:
                msg = ' Modified:'
            else:
                msg = (' Number of records differing from database but not '
                       'changed due to policy (use --overwrite to apply '
                       'changes):\n')
            self.stdout.write(msg + '\n'.join([
                '  {}: {}'.format(k, len(v))
                for k, v
                in changed.items()
            ]))
            if options['verbose_changes']:
                for m, i in changed.items():
                    for obj, change_list in i:
                        row = []
                        for old, new in change_list:
                            row.append('{} -> {}'.format(old, new))
                        line = '   {} {}: {}\n'.format(m, obj, ' | '.join(row))
                        self.stdout.write(line)

        else:
            self.stdout.write(' No records changed\n')

        self.stdout.write(' All done.')
