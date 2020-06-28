from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

from django.apps import apps
from django.db import transaction, IntegrityError

from .dataset import DATASET
from .models import (FecalSample, Note, Participant, Semester, Sequencing,
                     SequencingRun, Supplement, Week, Model)


class DryRunRollback(Exception):
    pass


class UserDataError(Exception):
    pass


class AbstractLoader():
    """
    Parent class for data importers

    Implementation needed for:
    COLS - a sequence of tuples, mapping column headers (as they appear in
           input file) to internal names; Column names will be matched
           casefolded.
    process_row() - method to import data from the current row/line, method
                    must be guarded by atomic transaction

    The dry_run option will have no effect on individual calls to
    process_line(), that are not calls via process_file(), only when
    process_file() is about to finish, dry_run will cause a rollback.
    """
    missing_data = ['']

    def __init__(self, colnames, sep='\t', can_overwrite=True,
                 warn_on_error=False, strict_sample_id=False, dry_run=False):
        # use internal names for columns:
        _cols = {i.casefold(): j for i, j in self.COLS}
        self.cols = []  # internal names for columns in file
        self.ignored_columns = []  # columns that won't be processed
        for i in colnames:
            if i.casefold() in _cols:
                self.cols.append(_cols[i.casefold()])
            else:
                self.cols.append(i)
                self.ignored_columns.append(i)

        self.warnings = []
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
    def load_file(cls, file, sep='\t', **kwargs):
        colnames = file.readline().strip().split(sep)
        loader = cls(colnames, sep=sep, **kwargs)
        return loader.process_file(file)

    def process_file(self, file):
        """
        Load data from given file
        """
        try:
            with transaction.atomic():
                for i in file:
                    self.process_line(i)

                if self.dry_run:
                    raise DryRunRollback
        except DryRunRollback:
            pass
        except UserDataError:
            # FIXME: needs to be reported; and (when) does this happen?
            raise
        except Exception as e:
            raise RuntimeError('Failed processing line:\n{}'.format(i)) from e

        return dict(
            count=self.count,
            new=self.new,
            added=self.added,
            changed=self.changed,
            ignored=self.ignored_columns,
            warnings=self.warnings,
            dry_run=self.dry_run,
            overwrite=self.can_overwrite,
        )

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

    def non_empty(self, value):
        """
        Say if a value is "empty" or missing.

        An empty value is something like whitespace-only or Null or None
        or 'NA' etc.
        """
        for i in self.missing_data:
            if isinstance(i, re.Pattern):
                if i.match(value):
                    return False
            elif value == i:
                return False
        if Model.decode_blank(value) == '':
            return False
        return True

    def process_line(self, line):
        """
        Process a single input line

        Calls process_row() which must be provided by implementors
        """
        if isinstance(line, str):
            row = [i.strip() for i in line.strip().split(self.sep)]

        # row: the non-empty row content, whitespace-stripped, read-only
        valid_cols = [i[1] for i in self.COLS]
        self.row = {
            k: v
            for k, v
            in zip(self.cols, row)
            if self.non_empty(v) and k in valid_cols
        }
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
            # some user errors in the data come up as IntegrityErrors, e.g.
            # violations of UNIQUE, IntegrityError should not be caught insie
            # an atomic() (cf. Django docs)
            msg1 = '{} at line {}'.format(e, self.count + 2)
            msg2 = ', current row:\n{}'.format(self.row)
            if self.warn_on_error \
                    and isinstance(e, (IntegrityError, UserDataError)):
                msg1 = '[SKIPPING]' + msg1
                self.warnings.append(msg1)
                print(msg1, file=sys.stderr)
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
            if 'supplement' in self.rec:
                from_row['supplement'] = self.rec['supplement']
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

    def process_supplement(self):
        """
        Process supplement related columns

        Call order: call before process_participant
        """
        from_row = self.get_from_row('frequency', 'dose', 'composition')

        if len(from_row) < 3:
            return
        if 'NA' in from_row.values():
            return

        obj, new = Supplement.objects.get_or_create(**from_row)
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
        ('Supplement_consumed', 'composition'),
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
        self.process_supplement()
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


class GeneralLoader(AbstractLoader):
    """
    Import data-set/model-specific file
    """
    def __init__(self, dataset, colnames, **kwargs):
        if dataset in DATASET:
            model_name = DATASET[dataset]['model']
        else:
            model_name = dataset

        self.conf = apps.get_app_config('mibios')
        self.model = self.conf.get_model(model_name)

        if dataset in DATASET:
            self.COLS = [
                (col, model_name + '__' + accs)
                for accs, col
                in DATASET[dataset]['fields']
            ]
            self.missing_data += DATASET[dataset].get('missing_data', [])
        else:
            # set COLS from model, start with id column
            self.COLS = [(model_name + '_id', model_name + '__id')]
            self.COLS += [
                (i.verbose_name.capitalize(), model_name + '__' + i.name)
                for i in self.model.get_simple_fields()
            ]

        super().__init__(colnames, **kwargs)

    @classmethod
    def load_file(cls, file, dataset=None, sep='\t', **kwargs):
        colnames = file.readline().strip().split(sep)
        loader = cls(dataset, colnames, sep=sep, **kwargs)
        return loader.process_file(file)

    def row_with_missing_columns(self):
        """
        Get row data with missing columns having an explicit blank

        DEPRECATED -- superceded by get_template etc.
        """
        row = self.row.copy()
        for col, accr in self.COLS:
            if accr not in row:
                row[accr] = ''
        return row

    def compile_template(self):
        """
        Make a deep template dict for the model(s)

        This replaces self.rec from super()
        """
        t = {}
        for i in self.row.keys():
            cur = t
            for j in i.split('__'):
                if j not in cur:
                    cur[j] = {}
                cur = cur[j]
        return t

    def tget(self, key):
        """
        Get method for dict for dicts / a.k.a. deep model template

        key can be a __-separated string (django lookup style) or a list
        of dict keys
        """
        cur = self.template
        if isinstance(key, str):
            key = key.split('__')

        for i in key:
            try:
                cur = cur[i]
            except (KeyError, TypeError):
                raise LookupError('Invalid key for template: {}'.format(key))
        return cur

    def tset(self, key, value):
        """
        Set method for dict for dicts / a.k.a. deep model template

        key can be a __-separated string (django lookup style) or a list
        of dict keys
        """
        cur = self.template
        prev = None
        if isinstance(key, str):
            key = key.split('__')

        for i in key:
            prev = cur
            try:
                cur = cur[i]
            except (KeyError, TypeError):
                raise LookupError('Invalid key for template: {}, current:'
                                  ''.format(key, cur))

        if not isinstance(cur, dict):
            # was assigned an object
            raise ValueError('Template already has a value:{} at key:{}'
                             ''.format(cur, key))
        prev[i] = value

    def process_row(self):
        self.template = self.compile_template()
        # Processing row column by column, i.e. iterating over accessors to the
        # template.  processing order: leafs go first, template roots (which
        # are not columns themselves) get added last
        a = sorted(
            self.row.items(),
            key=lambda x: x[0].split('__'),
            reverse=True
        ) + [(i, None) for i in self.template.keys()]
        for k, v in a:
            model, id_arg, obj, new = [None] * 4
            _k, _v, data = [None] *3
            try:
                parts = k.split('__')
                try:
                    # try as model
                    model = self.conf.get_model(parts[-1])
                except LookupError:
                    # assume a field
                    v = self.model.decode_blank(v)
                    # FIXME: are blanks not filtered out in process_line()?
                    if v:
                        self.tset(k, v)
                    continue

                # remove nodes not in row/data
                # TODO: if these are not leafs but "stand-ins" with some
                # fields filled in they get ignored currently
                data = self.tget(k).copy()
                for _k, _v in self.tget(k).items():
                    if isinstance(_v, dict):
                        del data[_k]

                if v:
                    id_arg = dict(canonical=v)
                    # add v as lookup to data so get_or_create() can create
                    # a new object if one does not yet exist
                    data.update(model.canonical_lookup(v))
                elif 'canonical' in data:
                    id_arg = dict(canonical=data.pop('canonical'))
                elif 'id' in data:
                    id_arg = dict(id=data.pop('id'))
                elif 'name' in data:
                    id_arg = dict(name=data.pop('name'))
                elif v is None:
                    # finally the primary row object but can't do
                    # anything with it?
                    # FIXME: should raise UserDataError if id or name
                    # column in empty, but have to find the right place
                    # where to make that determination
                    raise RuntimeError('oops here: data: ' + str(data))
                else:
                    # got nothing to id an object / remain a dict
                    continue

                # separate many_to_many fields from data
                m2ms = {
                    _k: _v
                    for _k, _v in data.items()
                    if model._meta.get_field(_k).many_to_many
                }
                for i in m2ms:
                    del data[i]

                obj, new = model.objects.get_or_create(
                    defaults=data,
                    **id_arg,
                )

                for _k, _v in m2ms.items():
                    getattr(obj, _k).add(_v)

                self.account(obj, new, data)
                self.tset(k, obj)
            except IntegrityError:
                raise
            except Exception as e:
                raise RuntimeError(
                    'k={} v={} model={} id_arg={}\ndata={}\ntemplate={}'
                    ''.format(k, v, model, id_arg, data, self.template)
                ) from e
            # if k == parts:
            #   print(f'X {k=} {v=} {model=} {id_arg=} {data=}' + 'template={}'
            #          ''.format(self.template))
            # else:
            #    print(parts)
