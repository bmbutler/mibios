from collections import Counter, defaultdict
from csv import DictReader, Sniffer
from inspect import signature
from io import TextIOBase, TextIOWrapper
import re
import sys

from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import transaction, IntegrityError

from . import get_registry
from .dataset import UserDataError
from .models import ImportFile, Model, NaturalKeyLookupError
from .utils import DeepRecord, getLogger


log = getLogger(__name__)
importlog = getLogger('dataimport')


class DryRunRollback(Exception):
    pass


class Loader():
    """
    Data importer
    """
    log = log
    dataset = None
    blanks = {None: ['']}

    def __init__(self, data_name, sep=None, can_overwrite=True,
                 warn_on_error=False, strict_sample_id=False, dry_run=False,
                 user=None, erase_on_blank=False):
        try:
            self.dataset = get_registry().datasets[data_name]
        except KeyError:
            self.model = get_registry().models[data_name]
        else:
            self.model = self.dataset.model

        model_name = self.model._meta.model_name
        if self.dataset:
            self.accr_map = {}
            for accr, col, *extra in self.dataset.fields:
                accr = model_name + '__' + accr
                self.accr_map[col.casefold()] = accr
                for i in extra:
                    if 'blanks' in i:
                        if accr not in self.blanks:
                            self.blanks[accr] = []
                        self.blanks[accr] += i['blanks']
        else:
            # set accessor map from model
            fields = self.model.get_fields(with_hidden=True)
            self.accr_map = {
                v.casefold(): model_name + '__' + n
                for v, n in zip(fields.verbose, fields.names)
            }
            if 'name' not in self.accr_map and hasattr(self.model, 'name'):
                self.accr_map['name'] = model_name + '__natural'

        self.warnings = []
        self.sep = sep
        self.new = Counter()
        self.added = Counter()
        self.changed = defaultdict(lambda: defaultdict(list))
        self.erased = defaultdict(lambda: defaultdict(list))
        self.count = 0
        self.fq_file_ids = set()
        self.can_overwrite = can_overwrite
        self.warn_on_error = warn_on_error
        self.strict_sample_id = strict_sample_id
        self.dry_run = dry_run
        self.user = user
        self.erase_on_blank = erase_on_blank
        self.file_record = None
        if dry_run:
            self.log = log
        else:
            self.log = importlog

        if self.dataset:
            self.blanks[None] += self.dataset.blanks

    def process_header(self):
        """
        Process the first row

        Helper for process_file()
        """
        self.ignored_columns = []  # columns that won't be processed
        for i in self.reader.fieldnames:
            if i.casefold() not in self.accr_map:
                self.ignored_columns.append(i)

        log.debug('accessor map:', self.accr_map)
        log.debug('ignored fields:', self.ignored_columns)

        if self.reader.fieldnames == self.ignored_columns:
            log.debug('input fields:', self.reader.fieldnames)
            raise UserDataError(
                'input file does not have any expected field/column names'
            )

    def pre_process_row(self, row):
        """
        Map file-fields to internal field names

        Remove fields not in spec, set blank fields to None
        Helper for process_row()
        """
        ret = {}
        for k, v in row.items():
            try:
                accessor = self.accr_map[k.casefold()]
            except KeyError:
                continue
            if self.is_blank(accessor, v):
                ret[accessor] = None
            else:
                ret[accessor] = v
        return ret

    def setup_reader(self, file):
        """
        Get the csv.DictReader all set up

        Helper for process_file()
        """
        if not isinstance(file, TextIOBase):
            # http uploaded files are binary
            file = TextIOWrapper(file)

        sniff_kwargs = {}
        reader_kwargs = {}
        if self.sep:
            sniff_kwargs['delimiters'] = self.sep
        try:
            dialect = Sniffer().sniff(file.read(5000), **sniff_kwargs)
        except Exception as e:
            log.debug('csv sniffer failed:', e)
            # trying fall-back (file might be too small)
            dialect = 'excel'  # set fall-bak default
            if self.sep:
                reader_kwargs['delimiter'] = self.sep
            else:
                reader_kwargs['delimiter'] = '\t'  # set fall-back default
        finally:
            file.seek(0)
            log.debug('sniffed:', vars(dialect))

        self.reader = DictReader(file, dialect=dialect, **reader_kwargs)
        self.sep = self.reader.reader.dialect.delimiter  # update if unset
        log.debug('delimiter:', '<tab>' if self.sep == '\t' else self.sep)
        log.debug('input fields:', self.reader.fieldnames)

    def process_file(self, file):
        """
        Load data from given file
        """
        log.debug('processing:', file, vars(file))
        self.linenum = 1
        self.last_warning = None
        row = None
        try:
            with transaction.atomic():
                self.file_record = ImportFile.create_from_file(file=file)
                # Saving input file to storage: if the input file come from the
                # local filesystem, ImportFile.save() we need to seek(0) our
                # file handle.  Do uploaded files in memory do something else?
                file.seek(0)
                # Getting the DictReader set up must happen after saving to
                # disk as csv.reader takes some sort of control over the file
                # handle and disabling tell() and seek():
                self.setup_reader(file)
                self.process_header()

                for row in self.reader:
                    self.process_row(row)

                if self.dry_run:
                    raise DryRunRollback
        except Exception as e:
            if self.file_record is not None:
                self.file_record.file.delete(save=False)
            if isinstance(e, DryRunRollback):
                pass
            elif isinstance(e, UserDataError):
                # FIXME: needs to be reported; and (when) does this happen?
                raise
            else:
                if row is None:
                    msg = 'error at file storage or opening stage'
                else:
                    msg = 'Failed processing line:\n{}'.format(row)
                raise RuntimeError(msg) from e

        return dict(
            count=self.count,
            new=self.new,
            added=self.added,
            changed=self.changed,
            erased=self.erased,
            ignored=self.ignored_columns,
            warnings=self.warnings,
            dry_run=self.dry_run,
            overwrite=self.can_overwrite,
            erase_on_blank=self.erase_on_blank,
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
        obj.add_change_record(
            file=self.file_record,
            line=self.linenum,
            user=self.user,
            cmdline=' '.join(sys.argv) if self.user is None else '',
        )
        need_to_save = False
        if is_new:
            self.new[model_name] += 1
            need_to_save = False
            obj.full_clean()
            obj.save()
        elif from_row is not None:
            consistent, diffs = obj.compare(from_row)
            for k, v in from_row.items():
                apply_change = False

                if k in diffs['only_them']:
                    apply_change = True
                    self.added[model_name] += 1
                elif k in diffs['only_us']:
                    self.erased[model_name][obj].append(
                        (k, getattr(obj, k))
                    )
                    if self.erase_on_blank:
                        apply_change = True
                elif k in diffs['mismatch']:
                    self.changed[model_name][obj].append(
                        (k, getattr(obj, k), from_row.get(k))
                    )
                    if self.can_overwrite:
                        apply_change = True

                if apply_change:
                    need_to_save = True
                    setattr(obj, k, v)

        if need_to_save:
            obj.full_clean()
            obj.save()

        self.rec[model_name] = obj

    def is_blank(self, col_name, value):
        """
        Say if a value is "empty" or missing.

        An empty value is something like whitespace-only or Null or None
        or 'NA' or equal to a specified blank value etc.

        Values are assumed to be trimmed of whitespace already.
        """
        for i in self.blanks[None] + self.blanks.get(col_name, []):
            if isinstance(i, re.Pattern):
                if i.match(value):
                    return True
            elif value == i:
                return True
        if Model.decode_blank(value) == '':
            return True
        return False

    def process_row(self, row):
        """
        Process a single input row

        This method does pre-processing and wraps the work into a transaction
        and handles some of the fallout of processing failure.  The actual work
        is delegated to process_fields().
        """
        self.linenum += 1
        self.row = self.pre_process_row(row)

        # rec: accumulates bits of processing before final assembly
        self.rec = {}
        # backup counters
        new_ = self.new
        added_ = self.added
        changed_ = self.changed
        erased_ = self.erased
        try:
            with transaction.atomic():
                self.process_fields()
        except (ValidationError, IntegrityError, UserDataError) as e:
            # Catch errors to be presented to the user;
            # some user errors in the data come up as IntegrityErrors, e.g.
            # violations of UNIQUE, IntegrityError should not be caught
            # inside an atomic() (cf. Django docs)
            if isinstance(e, ValidationError):
                msg = str(e.message_dict)
            else:
                msg = str(e)

            if not self.warn_on_error:
                # re-raise with row info added
                msg = 'at line {}: {}, current row:\n{}' \
                      ''.format(self.linenum, msg, self.row)
                raise type(e)(msg) from e

            # manage repeated warnings
            err_name = type(e).__name__
            if self.last_warning is None:
                self.last_warning = \
                        (err_name, msg, self.linenum, self.linenum)
            else:
                last_err, last_msg, first_line, last_line = \
                        self.last_warning
                if msg == last_msg and last_line + 1 == self.linenum:
                    # same warning as last line
                    self.last_warning = \
                            (last_err, msg, first_line, self.linenum)
                else:
                    # emit old warning
                    self.warnings.append(
                        'skipping row: at line {}: {} ({})'
                        ''.format(first_line, last_msg, last_err)
                    )
                    if last_line != first_line:
                        self.warnings.append(
                            '    (and for next {} lines)'
                            ''.format(last_line - first_line)
                        )
                    self.last_warning = \
                        (err_name, msg, self.linenum, self.linenum)
            # reset stats:
            self.new = new_
            self.added = added_
            self.changed = changed_
            self.erased = erased_
        except Exception as e:
            msg = 'at line {}: {}, current row:\n{}' \
                  ''.format(self.linenum, e, self.row)
            raise type(e)(msg) from e

        self.count += 1

    @classmethod
    def load_file(cls, file, data_name=None, **kwargs):
        loader = cls(data_name, **kwargs)
        return loader.process_file(file)

    def parse_value(self, accessor, value):
        """
        Delegate to specified parsing method
        """
        # rm model prefix from accsr to form method name
        pref, _, a = accessor.partition('__')
        parse_fun = getattr(self.dataset, 'parse_' + a, None)

        if parse_fun is None:
            ret = value
        else:
            args = [value]
            if len(signature(parse_fun).parameters) == 2:
                args.append(self.rec)

            try:
                ret = parse_fun(*args)
            except Exception as e:
                # assume parse_fun is error-free and blame user
                for i, j in self.accr_map.items():
                    if j == accessor:
                        col = i
                        break
                else:
                    col = '??'
                raise UserDataError(
                    'Failed parsing value "{}" in column {}: {}:{}'
                    ''.format(value, col, type(e).__name__, e)
                ) from e

        if isinstance(ret, dict):
            # put prefix back
            ret = {pref + '__' + k: v for k, v in ret.items()}

        return ret

    def get_model(self, accessor):
        """
        helper to get model class from accessor
        """
        name = accessor[0]
        m = get_registry().models[name]  # may raise KeyError

        for i in accessor[1:]:
            try:
                m = m._meta.get_field(i).related_model
            except (FieldDoesNotExist, AttributeError) as e:
                raise LookupError from e
            if m is None:
                raise LookupError
        return m

    def process_fields(self):
        """
        Process a row column by column

        Column representing non-foreign key fields are processed first and used
        to get/create their objects, root objects are created last
        """
        self.rec = DeepRecord()
        for k, v in self.row.items():
            if v is not None:
                v = self.parse_value(k, v)
                if v is None:
                    # parse_value said to ignore just this field
                    continue

            if isinstance(v, dict):
                self.rec.update(**v)
            else:
                self.rec[k] = v

        msg = 'line {}: record: {}'.format(self.linenum, self.rec)
        if self.dry_run:
            log.debug(msg)
        else:
            importlog.info(msg)

        for k, v in self.rec.items(leaves_first=True):
            model, id_arg, obj, new = [None] * 4
            _k, _v, data = [None] * 3
            try:
                try:
                    # try as model
                    model = self.get_model(k)
                except LookupError:
                    # assume a field
                    continue

                if isinstance(v, dict):
                    data = v.copy()
                    id_arg = {}
                elif isinstance(v, model):
                    # value was instantiated by parse_value()
                    # nothing left to do, skip accounting
                    continue
                elif v:
                    id_arg = dict(natural=v)
                    data = {}
                elif v is None:
                    # TODO: ?not get correct blank value for field?
                    continue
                else:
                    raise RuntimeError(
                        'oops here: data: {}\nk:{}\nv:{}\nstate:{}'
                        ''.format(data, k, v, self.rec)
                    )

                # separate identifiers from other fields
                for i in ['natural', 'id', 'name']:
                    if i in data:
                        id_arg[i] = data.pop(i)

                # separate many_to_many fields from data
                m2ms = {
                    _k: _v
                    for _k, _v in data.items()
                    if model._meta.get_field(_k).many_to_many
                }
                for i in m2ms:
                    del data[i]
                m2ms = {
                    _k: _v for _k, _v in m2ms.items()
                    # filter out Nones
                    if _v is not None
                }

                # ensure correct blank values
                data1 = {}
                for _k, _v in data.items():
                    if _v is None:
                        field = model._meta.get_field(_k)
                        if field.null:
                            data1[_k] = None
                        elif field.blank:
                            data1[_k] = ''
                        else:
                            # rm the field, will get default value for new objs
                            # TODO: issue a warning
                            continue
                    else:
                        data1[_k] = _v
                data = data1

                # if we don't have unique ids, use the "data" instead
                if not id_arg:
                    id_arg = data
                    data = {}

                try:
                    obj = model.objects.get(**id_arg)
                except model.DoesNotExist:
                    # id_arg was used as lookup in get() above but used now for
                    # the model constructor, this works as long as the keys are
                    # limited to field or property names
                    obj = model(**id_arg, **data)
                    new = True
                except model.MultipleObjectsReturned as e:
                    # id_arg under-specifies
                    msg = '{} is not specific enough for {}' \
                          ''.format(id_arg, model._meta.model_name)
                    raise UserDataError(msg) from e
                except NaturalKeyLookupError as e:
                    raise UserDataError(e) from e
                except ValueError as e:
                    # happens for value of wrong type, e.g. non-number in an id
                    # field so int() fails, and who knows, maybe other reasons,
                    # anyways, let's blame the user for uploading bad data.
                    raise UserDataError(
                        'Possibly bad value / type not matching the field: {}:'
                        '{}'.format(type(e).__name__, e)
                    ) from e
                else:
                    new = False

                self.account(obj, new, data)

                for _k, _v in m2ms.items():
                    getattr(obj, _k).add(_v)

                # replace with real object
                self.rec[k] = obj
            except (IntegrityError, UserDataError, ValidationError):
                raise
            except Exception as e:
                raise RuntimeError(
                    'k={} v={} model={} id_arg={}\ndata={}\nrow record=\n{}'
                    ''.format(k, v, model, id_arg, data,
                              self.rec.pretty(indent=(3, 2)))
                ) from e
