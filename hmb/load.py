from collections import Counter
from pathlib import Path
import re
import sys

from django.db import transaction

from .models import (FecalSample, Note, Participant, Semester, Sequencing,
                     SequencingRun, Week)


class UserDataError(Exception):
    pass


class AbstractLoader():
    """
    Parent class for data importers

    Implementation needed for:
    COLS - a sequence of tuples, mapping column headers to internal names;
           intended to allow quick adaption to changing input formats
    process_row() - method to import data from the current row/line, method
                    must be guarded by atomic transaction
    """
    def __init__(self, columns, sep='\t', add_only=True, warn_on_error=False,
                 strict_sample_id=False):
        valid_colnames = [i for i, j in self.COLS]
        for i in columns:
            if i not in valid_colnames:
                print('BORK', self.COLS)
                raise UserDataError('Unknown column name: {}'.format(i))
        # use internal names for columns:
        self.cols = [v for k, v in self.COLS if k in columns]
        self.sep = sep
        self.new = Counter()
        self.changed = Counter()
        self.count = 0
        self.fq_file_ids = set()
        self.add_only = add_only  # TODO: needs implementation; semantics?
        self.warn_on_error = warn_on_error
        self.strict_sample_id = strict_sample_id
        for col, name in self.COLS:
            setattr(self, 'name', None)

    @classmethod
    def load_file(cls, file, sep='\t', add_only=True, warn_on_error=False,
                  strict_sample_id=False):
        colnames = file.readline().strip().split(sep)
        loader = cls(colnames, sep=sep, add_only=add_only,
                     warn_on_error=warn_on_error,
                     strict_sample_id=strict_sample_id)
        return loader.process_file(file)

    @transaction.atomic
    def process_file(self, file):
        try:
            for i in file:
                self.process_line(i)
        except UserDataError:
            raise
        except Exception as e:
            raise RuntimeError('Failed processing line:\n{}'.format(i)) from e
        return self.count, self.new, self.changed

    def process_line(self, line):
        """
        Process a single input line

        Calls process_row() which must be provided by implementors
        """
        if type(line) == str:
            row = line.strip().split(self.sep)

        # row and rec keep processing state for current line
        # only non-empty fields are present in the row dict
        self.row = {k: v.strip() for k, v in zip(self.cols, row) if v.strip()}
        self.rec = {}
        # backup counters
        new_ = self.new
        changed_ = self.changed
        try:
            self.process_row()
        except Exception as e:
            msg1 = '{} at line {}'.format(e, self.count + 2)
            msg2 = ', current row:\n{}'.format(self.row)
            if self.warn_on_error and isinstance(e, UserDataError):
                print('[SKIPPING]', msg1, file=sys.stderr)
                self.new = new_
                self.changed = changed_
            else:
                # re-raise with row into added
                raise type(e)(msg1 + msg2) from e

        self.count += 1

    def process_semester(self):
        """
        Convert semester from input to object
        """
        if 'semester' in self.row:
            try:
                s = Semester.parse(self.row['semester'])
            except ValueError as e:
                raise UserDataError(str(e)) from e
            self.rec['semester'], new = Semester.objects.get_or_create(**s)
            if new:
                self.new['semester'] += 1

    def process_participant(self):
        """
        Convert input participant into object

        Call order: call after process_semester()
        """
        if 'participant' in self.row:
            obj, new = Participant.objects.get_or_create(
                name=self.row['participant']
            )
            if 'semester' in self.rec:
                obj.semester = self.rec['semester']
                obj.save()
            if new:
                self.new['participant'] += 1
            self.rec['participant'] = obj

    def process_week(self):
        """
        Convert input seek into object
        """
        if 'week' in self.row:
            self.rec['week'], new = Week.objects.get_or_create(
                **Week.parse(self.row['week'])
            )
            if new:
                self.new['week'] += 1

    def process_sample_id(self):
        """
        Convert input sample_id into sample object

        Call order: call after process_participant
        """
        if 'sample_id' in self.row:
            try:
                sample_id = FecalSample.parse_id(self.row['sample_id'])
            except ValueError as e:
                if self.strict_sample_id:
                    raise UserDataError(str(e)) from e
                else:
                    return
            if 'participant' in self.rec:
                if sample_id['participant'] != self.rec['participant'].name:
                    raise UserDataError(
                        'Participant and Sample IDs inconsistent'
                    )
                sample_id['participant'] = self.rec['participant']
            fecal_sample, new = FecalSample.objects.get_or_create(
                **sample_id,
            )

            if new:
                self.new['fecal sample'] += 1

            if 'week' in self.row:
                fecal_sample.week = self.rec['week']
                fecal_sample.save()

            self.rec['fecal_sample'] = fecal_sample


class SampleMasterLoader(AbstractLoader):
    """
    Loader for Robert's sample master file
    """
    COLS = [
        ('sample', 'fq_file_id'),
        ('participant', 'participant'),
        ('control', 'is_control'),
        ('control group', 'control_class'),
        ('do not use', 'note'),
        ('R1fastq', 'r1'),
        ('R2fastq', 'r2'),
    ]

    @transaction.atomic
    def process_row(self):
        self.row['fq_file_id'] = self.row['fq_file_id'].replace('-', '_')
        self.process_participant()
        self.process_sample_id()
        r1 = Path(self.row['r1'])
        r2 = Path(self.row['r2'])

        if 'control' in self.row:
            try:
                self.rec['control'] = \
                    Sequencing.parse_control(self.row['control'])
            except ValueError as e:
                raise UserDataError(str(e)) from e

        o, new = Sequencing.objects.get_or_create(name=self.row['fq_file_id'])
        if new:
            self.new['seq sample'] += 1
        else:
            if 'fecal_sample' in self.rec:
                if o.sample != self.rec['fecal_sample']:
                    self.changed['sequencing'] += 1
        if 'fecal_sample' in self.rec:
            o.sample = self.rec['fecal_sample']

        if 'control' in self.rec:
            o.control = self.rec['control']
        o.r1_file = r1
        o.r2_file = r2
        o.save()


class SequencingLoader(AbstractLoader):
    """
    Loader for data in "meta data format" tables
    """
    # COLS: map column names to internal names
    COLS = (
        ('FASTQ_ID', 'fq_file_id'),
        ('Participant_ID', 'participant'),
        ('Sample_ID', 'sample_id'),
        ('Study_week', 'week'),
        ('Semester', 'semester'),
        ('Use_Data', 'can_use'),
        ('Quantity_compliant', 'qcomp'),
        ('Frequency', 'freq'),
        ('Total_dose_grams', 'dose'),
        ('Supplement_consumed', 'supp'),
        ('pH', 'ph'),
        ('Bristol', 'bristol'),
        ('seq_serial', 'serial'),
        ('seq_run', 'run'),
        ('drop', 'note'),
    )

    @transaction.atomic
    def process_row(self):
        self.process_semester()
        self.process_participant()
        self.process_week()
        self.process_sample_id()

        self.rec['run'], new = SequencingRun.objects.get_or_create(
            serial=self.row['serial'], number=self.row['run']
        )
        if new:
            self.new['runs'] += 1

        o, new = Sequencing.objects.get_or_create(name=self.row['fq_file_id'])
        if new:
            self.new['seq sample'] += 1
        else:
            if o.sample != self.rec['fecal_sample']:
                self.changed['sequencing'] += 1
        o.sample = self.rec['fecal_sample']
        o.save()

        # TODO: shall notes be deleted?
        if 'note' in self.row:
            note, new = Note.objects.get_or_create(name=self.row['note'])
            o.note.add(note)
            if new:
                self.new['note'] += 1
            self.rec['note'] = note


class MMPManifestLoader(AbstractLoader):
    """
    Loader for Jonathans MMP_Manifest file
    """
    # map table header to internal names
    COLS = [
        ('specimen', 'specimen'),
        ('batch', 'batch'),
        ('R1', 'R1'),
        ('R2', 'R2'),
        ('person', 'participant'),
        ('Sample_ID', 'sample_id'),
        ('semester', 'semester'),
        ('plate', 'plate'),
        ('seqlabel', 'snum'),
        ('read__1_fn', 'read__1_fn'),
        ('read__2_fn', 'read__2_fn'),
    ]

    plate_pat = re.compile(r'^P([0-9])-([A-Z][0-9]+)$')
    snum_pat = re.compile(r'^S([0-9]+$)')

    def process_sample_id(self):
        if 'sample_id' in self.row:
            self.row['sample_id'] = self.row['sample_id'].replace('-', '_')
            super().process_sample_id()

    @transaction.atomic
    def process_row(self):
        self.process_semester()
        self.process_participant()
        self.process_sample_id()

        # 1. get fastq paths
        r1 = Path(self.row['R1'])
        r2 = Path(self.row['R2'])
        # 2. cut for "sample id" after S-number
        fq_file_id = re.sub(r'(_S[0-9]+).*', r'\1', r1.name)
        # sanity check
        if not r2.name.startswith(fq_file_id):
            raise UserDataError('fastq file name inconsistency')
        # make mothur compatible
        fq_file_id = fq_file_id.replace('-', '_')
        # 3.parse plate+position
        if 'plate' in self.row:
            m = self.plate_pat.match(self.row['plate'])
            if m is None:
                raise UserDataError('Failed parsing plate field')
            plate, position = m.groups()
            plate = int(plate)
        # snum
        if 'snum' in self.row:
            snum = self.row['snum']
            if not fq_file_id.endswith(snum):
                raise UserDataError('S-number is inconsistent with filenames')
            m = re.match(self.snum_pat, snum)
            if m is None:
                raise UserDataError('Failed parsing s-number')
            snum = int(m.groups()[0])

        o, new = Sequencing.objects.get_or_create(name=fq_file_id)
        if new:
            self.new['seq samples'] += 1
        else:
            if 'fecal_sample' in self.rec:
                if o.sample != self.rec['fecal_sample']:
                    self.changed['sequencing'] += 1
        if 'fecal_sample' in self.rec:
            o.sample = self.rec['fecal_sample']
        o.r1_file = str(r1)
        o.r2_file = str(r2)
        if 'plate' in self.row:
            o.plate = plate
            o.position = position
        if 'snum' in self.row:
            o.snumber = snum
        o.save()
