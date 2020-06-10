from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

from django.db import transaction

from .models import (Diet, FecalSample, Note, Participant, Semester,
                     Sequencing, SequencingRun, Week)


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

    The dry_run option will have no effect on individual calls to
    process_line(), that are not calls via process_file(), only when
    process_file() is about to finish, dry_run will cause a rollback.
    """
    def __init__(self, columns, sep='\t', can_overwrite=True,
                 warn_on_error=False, strict_sample_id=False, dry_run=False):
        valid_colnames = [i for i, j in self.COLS]
        for i in columns:
            if i not in valid_colnames:
                raise UserDataError('Unknown column name: {}'.format(i))
        # use internal names for columns:
        self.cols = [v for k, v in self.COLS if k in columns]
        self.sep = sep
        self.new = Counter()
        self.added = Counter()
        self.changed = defaultdict(list)
        self.count = 0
        self.fq_file_ids = set()
        self.can_overwrite = can_overwrite
        self.warn_on_error = warn_on_error
        self.strict_sample_id = strict_sample_id
        self.dry_run = dry_run
        for col, name in self.COLS:
            setattr(self, 'name', None)

    @classmethod
    def load_file(cls, file, sep='\t', can_overwrite=True, warn_on_error=False,
                  strict_sample_id=False, dry_run=False):
        colnames = file.readline().strip().split(sep)
        loader = cls(colnames, sep=sep, can_overwrite=can_overwrite,
                     warn_on_error=warn_on_error, dry_run=dry_run,
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
        if self.dry_run:
            transaction.rollback()
        return self.count, self.new, self.added, self.changed

    def get_from_row(self, *keys):
        """
        Get a dict with specified keys based on row

        Helper method to update object templates
        """
        ret = {}
        for i in keys:
            if i in self.row:
                ret[i] = self.row[i]
        return ret

    def account(self, obj, is_new, from_row=None):
        """
        Account for object creation, change

        Enforce object overwrite as needed.
        Update state with object
        """
        model_name = obj._meta.model_name
        if is_new:
            self.new[model_name] += 1
        elif from_row is not None:
            consistent, diff = obj.compare(from_row)
            if diff:
                if consistent:
                    self.added[model_name] += 1
                else:
                    self.changed[model_name].append((
                        obj,
                        [
                            (getattr(obj, i), from_row.get(i))
                            for i in diff
                        ]
                    ))

                if consistent or self.can_overwrite:
                    for k, v in from_row.items():
                        setattr(obj, k, v)
                    obj.save()

        self.rec[model_name] = obj

    def process_line(self, line):
        """
        Process a single input line

        Calls process_row() which must be provided by implementors
        """
        if type(line) == str:
            row = line.strip().split(self.sep)

        # row: the non-empty row content, whitespace-stripped, read-only
        self.row = {k: v.strip() for k, v in zip(self.cols, row) if v.strip()}
        # rec: accumulates bits of processing before final assembly
        self.rec = {}
        # backup counters
        new_ = self.new
        added_ = self.added
        changed_ = self.changed
        try:
            with transaction.atomic():
                self.process_row()
        except Exception as e:
            msg1 = '{} at line {}'.format(e, self.count + 2)
            msg2 = ', current row:\n{}'.format(self.row)
            if self.warn_on_error and isinstance(e, UserDataError):
                print('[SKIPPING]', msg1, file=sys.stderr)
                self.new = new_
                self.added = added_
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
                s = Semester.canonical_lookup(self.row['semester'])
            except ValueError as e:
                raise UserDataError(str(e)) from e
            obj, new = Semester.objects.get_or_create(**s)
            self.account(obj, new)

    def process_participant(self):
        """
        Convert input participant into object

        Call order: call after process_semester()
        """
        if 'participant' in self.row:
            from_row = self.get_from_row('quantity_compliant')
            from_row['name'] = self.row['participant']
            if 'diet' in self.rec:
                from_row['diet'] = self.rec['diet']
            if 'semester' in self.rec:
                from_row['semester'] = self.rec['semester']
            obj, new = Participant.objects.get_or_create(
                defaults=from_row,
                name=from_row['name']
            )
            self.account(obj, new, from_row)

    def process_week(self):
        """
        Convert input week into object
        """
        if 'week' in self.row:
            obj, new = Week.objects.get_or_create(
                **Week.canonical_lookup(self.row['week'])
            )
            self.account(obj, new)

    def process_sample_id(self, from_row={}):
        """
        Convert input sample_id into sample object

        Call order: call after process_participant
        """
        if 'sample_id' in self.row:
            try:
                from_row.update(FecalSample.parse_id(self.row['sample_id']))
            except ValueError as e:
                if self.strict_sample_id:
                    raise UserDataError(str(e)) from e
                else:
                    return
            if 'participant' in self.rec:
                if from_row['participant'] != self.rec['participant'].name:
                    raise UserDataError(
                        'Participant and Sample IDs inconsistent'
                    )
                from_row['participant'] = self.rec['participant']

            if 'week' in self.row:
                from_row['week'] = self.rec['week']

            obj, new = FecalSample.objects.get_or_create(
                defaults=from_row,
                participant=from_row['participant'],
                number=from_row['number'],
            )
            self.account(obj, new, from_row)

    def process_note(self):
        if 'note' in self.row:
            obj, new = Note.objects.get_or_create(name=self.row['note'])
            self.account(obj, new)

    def process_diet(self):
        """
        Process diet related columns

        Call order: call before process_participant
        """
        from_row = self.get_from_row('frequency', 'dose', 'supplement')

        if len(from_row) < 3:
            return
        if 'NA' in from_row.values():
            return

        obj, new = Diet.objects.get_or_create(**from_row)
        self.account(obj, new)


class SampleMasterLoader(AbstractLoader):
    """
    Loader for Robert's sample master file
    """
    COLS = [
        ('sample', 'fq_file_id'),
        ('participant', 'participant'),  # ignore
        ('control', 'control'),
        ('control group', 'control_group'),
        ('do not use', 'note'),
        ('R1fastq', 'r1'),
        ('R2fastq', 'r2'),
    ]

    def process_row(self):
        self.row['fq_file_id'] = self.row['fq_file_id'].replace('-', '_')
        self.process_note()

        # template to use for create or comparison with existing object:
        from_row = {}
        from_row['name'] = self.row['fq_file_id']
        from_row['r1_file'] = str(Path(self.row['r1']))
        from_row['r2_file'] = str(Path(self.row['r2']))

        if 'control' in self.row:
            try:
                from_row['control'] = \
                    Sequencing.parse_control(self.row['control'])
            except ValueError as e:
                raise UserDataError(str(e)) from e

        obj, new = Sequencing.objects.get_or_create(
            defaults=from_row,
            name=from_row['name']
        )
        self.account(obj, new, from_row)

        if 'note' in self.rec:
            obj.note.add(self.rec['note'])


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
        ('Use_Data', 'use_data'),
        ('Quantity_compliant', 'quantity_compliant'),
        ('Frequency', 'frequency'),
        ('Total_dose_grams', 'dose'),
        ('Supplement_consumed', 'supplement'),
        ('pH', 'ph'),
        ('Bristol', 'bristol'),
        ('seq_serial', 'serial'),
        ('seq_run', 'run'),
        ('drop', 'note'),
    )

    def process_sample_id(self, from_row={}):
        from_row.update(self.get_from_row(
            'ph',
            'bristol',
        ))
        # TODO: verify meaning on NA
        for i in ['ph', 'bristol']:
            if from_row[i] == 'NA':
                del from_row[i]
        super().process_sample_id(from_row)

    def process_row(self):
        if 'use_data' in self.row:
            if not self.row['use_data'].lower() == 'yes':
                raise UserDataError('encountered use_data!=yes')
        self.process_semester()
        self.process_diet()
        self.process_participant()
        self.process_week()
        self.process_sample_id()
        self.process_note()

        # template to use for create or comparison with existing object:
        from_row = {}
        from_row['name'] = self.row['fq_file_id']
        from_row['sample'] = self.rec['fecalsample']
        if 'run' in self.row:
            run, new = SequencingRun.objects.get_or_create(
                serial=self.row['serial'],
                number=self.row['run'],
            )
            self.account(run, new)
            from_row['run'] = run

        obj, new = Sequencing.objects.get_or_create(
            defaults=from_row,
            name=from_row['name'],
        )
        self.account(obj, new, from_row)

        # One note may get added here but existing notes not removed
        if 'note' in self.rec:
            obj.note.add(self.rec['note'])


class MMPManifestLoader(AbstractLoader):
    """
    Loader for Jonathans MMP_Manifest file
    """
    # map table header to internal names
    COLS = [
        ('specimen', 'specimen'),
        ('batch', 'batch'),  # ignore, use plate
        ('R1', 'r1'),
        ('R2', 'r2'),
        ('person', 'participant'),
        ('Sample_ID', 'sample_id'),
        ('semester', 'semester'),
        ('plate', 'plate'),
        ('seqlabel', 'snum'),
        ('read__1_fn', 'read__1_fn'),  # ignore
        ('read__2_fn', 'read__2_fn'),  # ignore
    ]

    plate_pat = re.compile(r'^P([0-9])-([A-Z][0-9]+)$')
    snum_pat = re.compile(r'^S([0-9]+$)')

    def process_sample_id(self):
        if 'sample_id' in self.row:
            self.row['sample_id'] = self.row['sample_id'].replace('-', '_')
            super().process_sample_id()

    def process_row(self):
        self.process_semester()
        self.process_participant()
        self.process_sample_id()

        # template to use for create or comparison with existing object:
        from_row = {}

        # 1. get fastq paths
        from_row['r1_file'] = str(Path(self.row['r1']))
        from_row['r2_file'] = str(Path(self.row['r2']))
        # 2. cut for "sample id" after S-number
        fq_file_id = re.sub(r'(_S[0-9]+).*', r'\1', Path(self.row['r1']).stem)
        # sanity check
        if not Path(from_row['r2_file']).name.startswith(fq_file_id):
            raise UserDataError('fastq file name inconsistency')
        # make mothur compatible
        fq_file_id = fq_file_id.replace('-', '_')
        from_row['name'] = fq_file_id
        # 3.parse plate+position
        if 'plate' in self.row:
            m = self.plate_pat.match(self.row['plate'])
            if m is None:
                raise UserDataError('Failed parsing plate field')
            plate, position = m.groups()
            plate = int(plate)
            from_row['plate'] = plate
            from_row['plate_position'] = position
        # snum
        if 'snum' in self.row:
            snum = self.row['snum']
            if not fq_file_id.endswith(snum):
                raise UserDataError('S-number is inconsistent with filenames')
            m = re.match(self.snum_pat, snum)
            if m is None:
                raise UserDataError('Failed parsing s-number')
            from_row['snumber'] = int(m.groups()[0])
        if 'fecalsample' in self.rec:
            from_row['sample'] = self.rec['fecalsample']

        obj, new = Sequencing.objects.get_or_create(
            defaults=from_row,
            name=from_row['name'],
        )
        self.account(obj, new, from_row)
