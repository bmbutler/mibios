import argparse

from django.core.management.base import BaseCommand, CommandError

from mibios.export import to_meta_csv


class Command(BaseCommand):
    help = 'Export data to meta file format'

    def add_arguments(self, argp):
        argp.add_argument(
            'file',
            type=argparse.FileType('w'),
            help='Name of output file'
        )
        argp.add_argument(
            '-s', '--sep',
            default='\t',
            help='Column separator, default is <tab>',
        )

    def handle(self, *args, **options):
        try:
            to_meta_csv(file=options['file'], sep=options['sep'])
        except Exception as e:
            raise CommandError(e) from e
