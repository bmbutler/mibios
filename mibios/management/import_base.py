import argparse

from django.core.management.base import BaseCommand, CommandError

from mibios.load import UserDataError
from mibios.utils import getLogger


class AbstractImportCommand(BaseCommand):
    # Implementers must loader class, which must be
    # an implementation of mibios.load.AbstractLoader
    loader_class = None

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
            '--erase-on-blank',
            action='store_true',
            help='Erase fields in the database if the input file has the '
                 'corresponding column but the field is empty/blank.  Which '
                 'values are considered blank depends on the table and column.'
                 ' By default existing data is not touched.  This option is '
                 'separate from --overwrite and only affects the case when an '
                 'empty value overwrites a non-empty while --overwrite allows '
                 'to overwrite a non-empty value with a non-empty.'
        )
        parser.add_argument(
            '--no-new-records',
            action='store_true',
            help='Allow existing records to be changed but do not create new '
                 'records.  Rows for which no exiting record is found either '
                 'raise a warning or an error depending on the --warn-on-error'
                 ' option.',
        )
        parser.add_argument(
            '-s', '--sep',
            default=None,
            help='Column separator, if not given the script will auto-detect '
                 'and if that fails default to <tab>',
        )
        parser.add_argument(
            '--verbose-changes',
            action='store_true',
            help='List each change of existing values, the default is to just '
                 'give a summary',
        )
        parser.add_argument(
            '--warn-on-error',
            action='store_true',
            help='Skip a row and receive a warning if processing it results '
                 'in an error.  By default the import is aborted on any such '
                 'error.',
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Turn on debugging output'
        )
        return parser

    def handle(self, *args, **options):
        logger = getLogger('dataimport')
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
                erase_on_blank=options['erase_on_blank'],
                dry_run=options['dry_run'],
                warn_on_error=options['warn_on_error'],
                no_new_records=options['no_new_records'],
                **self.load_file_kwargs(**options),
            )
        except UserDataError as e:
            raise CommandError(e)
        except Exception as e:
            raise CommandError('Failed importing data') from e

        kwargs = dict(**options)
        kwargs.update(**stats)
        log_msg = self.format_import_stats(**kwargs)

        file_rec = stats.get('file_record', None)
        if file_rec is not None:
            file_rec.log = log_msg
            file_rec.save()

        logger.info(log_msg)
        self.stdout.write(' All done.')

    @classmethod
    def format_import_stats(cls, count=0, new={}, added={}, changed={},
                            ignored=[], warnings=[], erased={}, **options):
        out = ' Options:\n  '
        out += '\n  '.join([
            str(k) if v is True else '{}: {}'.format(k, v)
            for k, v in options.items()
            if v
        ]) + '\n'
        out += ' Rows processed: {}\n'.format(count)
        if ignored:
            out += ' Column(s) not processd ({}): '.format(len(ignored))
            out += ', '.join(ignored) + '\n'
        if new:
            out += ' Imported:\n'
            for k, v in new.items():
                out += f'  {k} ({len(v)}):\n'
                out += ''.join([f'    {i}\n' for i in v])
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
                    for obj, change_list in i.items():
                        row = []
                        for field, old, new in change_list:
                            row.append('{}: {} -> {}'.format(field, old, new))
                        out += '   {} {}: {}\n'.format(m, obj, ' | '.join(row))

        if erased:
            if options.get('erase_on_blank'):
                msg = ' Erased:\n'
            else:
                msg = (' Blank values: (use "erase-on-blank" option to erase):'
                       '\n')
            out += msg + '\n'.join([
                '  {}: {}'.format(k, len(v))
                for k, v
                in erased.items()
            ]) + '\n'
            if options.get('erase_on_blank') \
                    and options.get('verbose_changes'):
                for m, m_items in erased.items():
                    for obj, field_values in m_items.items():
                        row = []
                        for field, val in field_values:
                            row.append('{}: {}'.format(field, val))
                        out += '   {} {}: {}\n'.format(m, obj, ' | '.join(row))

        if not changed and not erased:
            out += ' No existing records modified\n'

        if warnings:
            out += ' Warnings ({}):\n'.format(len(warnings))
            for i in warnings:
                out += '  {}\n'.format(i)

        return out
