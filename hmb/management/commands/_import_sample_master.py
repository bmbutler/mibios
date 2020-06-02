import argparse

from django.db import transaction
from django.core.management.base import BaseCommand, CommandError

from hmb import modules as m


class Command(BaseCommand):
    help = 'Import data from "sample master file"'

    COLS = ['sample', 'participant', 'control', 'control group', 'do not use',
            'R1fastq', 'R2fastq']

    def add_Argument(self, argp):
        argp.add_argument(
            'file',
            type=argparse.FileType(),
            help='Inputfile, tab-separated with columns: '
                 '' + ', '.join(self.COLS),
        )

    @transaction.atomic
    def handle(self, *args, **options):
        head = options['file'].readline().strip().split('\t')
        if head != COLS:
            raise CommandError('Table header of input file not recognized')

        for line in options['file']:
            row = line.strip().split('\t')
            sample, subj, control, cgroup, nouse, r1, r2 = row
            if nouse == 'no consent':
                continue
            
            ...
