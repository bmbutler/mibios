from datetime import datetime
from functools import partial, wraps
from inspect import signature
from itertools import zip_longest
from operator import length_hint
import os
from pathlib import Path
from threading import local, Timer
from string import Formatter
import sys

import pandas

from django.db import router, transaction

# Workaround for weird pandas/xlrd=1.2/defusedxml combination runtime issue,
# we'll get an AttributeError: 'ElementTree' object has no attribute
# 'getiterator' inside xlrd when trying to pandas.read_excel().  See also
# https://stackoverflow.com/questions/64264563
import xlrd
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True


thread_data = local()
thread_data.timer = None


def get_last_timer():
    return thread_data.timer


class ReturningGenerator:
    """
    A wrapper to catch return values of generators

    Usage:
    def g(*args):
        ...
        yield ...
        ...
        return x

    g = ReturningGenerator(g())
    for i in g:
        do_stuff()
    foo = g.value

    An AttributeError will be raised if one attempts to access the return value
    before the generator is exhausted.
    """
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        self.value = yield from self.generator


class RepeatTimer(Timer):
    """
    Run given function repeatedly and wait a given interval between invocations

    From an answer to stackoverflow.com/questions/12435211
    """
    def __init__(self, *args, owner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner
        # let main thread exit if we're forgotten and never stop ticking
        self.daemon = True

    def start(self):
        thread_data.timer = self
        super().start()

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class ProgressPrinter():
    """
    A simple timer-based printer of progress or counts

    How to use:

    pp = ProgressPrinter('{progress} foo done')
    count = 0
    for i in some_iterator():
        do_stuff()
        count += 1
        pp.update(count)
    pp.finished()

    This will print "<n> foo done" once per second to the terminal and print
    the final count after the for loop ends. The internal timer will keep
    restarting as long as the update() method is called with changing progress
    counts.

    After the timer has stopped, even after calling finish() the progress
    printing can be resumed by calling update() with a different state than the
    last one.
    """
    DEFAULT_TEMPLATE = '{progress}'
    DEFAULT_INTERVAL = 1.0  # seconds

    def __init__(
            self,
            template=DEFAULT_TEMPLATE,
            interval=DEFAULT_INTERVAL,
            output_file=sys.stdout,
            show_rate=True,
            length=None,
    ):
        if interval <= 0:
            raise ValueError('interval must be greater than zero')

        self.template, self.template_var = self._init_template(template)
        self.interval = interval
        self.output_file = output_file
        self.show_rate = show_rate
        self.to_terminal = output_file.isatty()
        self.length = length
        self.it = None
        self.timer = None
        # start metering here, assuming inc() calls commence soon:
        self.reset_state()
        self.timer.start()

    def __call__(self, it):
        self.it = it
        self.reset_length_info()
        for elem in it:
            yield elem
            self.inc()
        self.it = None
        self.finish()

    def reset_state(self):
        """
        reset the variable state

        Must be called before inc().  Will start the timer and progress
        metering and printing.
        """
        self.reset_timer()
        self.current = 0
        self.last = 0
        self.at_previous_ring = None
        self.max_width = 0
        self.ring_time = None
        self.time_zero = datetime.now()
        self.reset_length_info()

    def reset_length_info(self):
        """
        get length if possible
        """
        self._length = None
        self.file_size = None
        if self.length is None:
            try:
                self.file_size = os.stat(self.it.fileno()).st_size
            except Exception:
                self.file_size = None

                hint = length_hint(self.it)
                if hint > 0:
                    # 0 is the default in case there is no length or length
                    # hint, it seems we can't tell this from an actual length
                    # of zero
                    self._length = hint
        else:
            self._length = self.length

    def reset_timer(self):
        """
        Initializes or resets the timer but does not start() it.
        """
        if self.timer is not None:
            self.timer.cancel()
        self.timer = RepeatTimer(self.interval, self._ring, owner=self)

    def _init_template(self, template):
        """
        set up template

        We support templates with zero or one formatting fields, the single
        field may be named or anonymous.
        """
        fmt_vars = [
            i[1] for i
            in Formatter().parse(template)
            if i[1] is not None
        ]
        if len(fmt_vars) == 0:
            template = '{} ' + template
            template_var = None
        elif len(fmt_vars) == 1:
            # keep tmpl as-is
            if fmt_vars[0] == '':
                template_var = None
            else:
                template_var = fmt_vars[0]
        else:
            raise ValueError(f'too many format fields in template: {fmt_vars}')

        return template, template_var

    def inc(self, step=1):
        """
        increment progress

        Turn on time if needed
        """
        if self.current == self.at_previous_ring:
            # timer was stopped at last ring
            try:
                self.timer.start()
            except RuntimeError:
                # if the timer thread is still in reset_timer(), we can't
                # re-start the old timer thread; pass here and start new timer
                # at next inc()
                pass

        self.last = self.current
        self.current += step

    def finish(self):
        """ Stop the timer and print a final result """
        total_seconds = (datetime.now() - self.time_zero).total_seconds()
        avg_txt = (f'(total: {total_seconds:.1f}s '
                   f'avg: {self.current / total_seconds:.1f}/s)')
        self.print_progress(avg_txt=avg_txt, end='\n')  # print with totals/avg
        self.reset_state()

    def _ring(self):
        """ Print progress """
        self.ring_time = datetime.now()
        self.print_progress()

        if self.current == self.at_previous_ring:
            # maybe we just iterate very slowly relative to the timer interval,
            # but probably some exception occurred in the main thread; have to
            # stop the timer or we'll get an infinite loop
            # When inc() is called again a new timer will be used.
            self.reset_timer()
            return

        self.at_previous_ring = self.current

    def estimate(self):
        """
        get current percentage and estimated finish time

        Returns None if we havn't made any progress yet or we don't know the
        length of the iterator.
        """
        if self.current == 0:
            # too early for estimates (and div by zero)
            return

        if self.file_size is not None and self.file_size > 0:
            # best effort to get stream position, hopefully in bytes
            try:
                pos = self.it.tell()
            except Exception:
                # for iterating text io tell() is diabled?
                # (which doesn't seem to be documented?)
                try:
                    pos = self.it.buffer.tell()
                except Exception:
                    return
            frac = pos / self.file_size

        elif self._length is not None and self._length > 0:
            frac = self.current / self._length
        else:
            # no length info
            return

        cur_duration = (self.ring_time - self.time_zero).total_seconds()
        remain = cur_duration / frac - cur_duration
        return frac, remain

    def print_progress(self, avg_txt='', end=''):
        """ Do the progress printing """
        if self.template_var is None:
            txt = self.template.format(self.current)
        else:
            txt = self.template.format(**{self.template_var: self.current})

        if avg_txt:
            # called by finish()
            txt += ' ' + avg_txt
        elif self.show_rate:
            prev = self.at_previous_ring
            if prev is None:
                prev = 0
            try:
                rate = (self.current - prev) / self.interval
            except Exception:
                # math errors if current is not a number?
                pass
            else:
                est = self.estimate()
                if est is None:
                    # just the rate
                    txt += f' ({rate:.0f}/s)'
                else:
                    frac, remain = est
                    txt += f' ({frac:.0%} rate:{rate:.0f}/s remaining:{remain:.0f}s)'  # noqa:E501

        self.max_width = max(self.max_width, len(txt))
        txt = txt.ljust(self.max_width)

        if self.to_terminal:
            txt = '\r' + txt

        print(txt, end=end, flush=True, file=self.output_file)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into non-overlapping groups of n elements

    This is the grouper from the stdlib's itertools' recipe book.
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def chunker(iterable, n):
    """
    Group iterable in chunks of equal size, except possibly for the last chunk
    """
    sentinel = object()
    for grp in grouper(iterable, n, fillvalue=sentinel):
        # grp is a n-tuple
        if grp[-1] is sentinel:
            yield tuple(i for i in grp if i is not sentinel)
        else:
            yield grp


class InputFileSpec:
    IGNORE_COLUMN = object()
    SKIP_ROW = object()
    CALC_VALUE = object()

    empty_values = []
    """
    A list of input-file-wide extra empty values.  For the purpose of loading
    the data these are used in addition to each field's empty_values attribute.
    """

    def __init__(self, *column_specs):
        self._spec = column_specs or None

        # set by setup():
        self._fields = None
        self._convfuncs = None
        self.model = None
        self.loader = None
        self.path = None
        self.has_header = None
        self.fk_attrs = {}

    def setup(self, loader, column_specs=None, path=None):
        """
        Setup method to be called once before loading data

        Intended to be called automatically by the loader.
        """
        self.loader = loader
        self.model = loader.model
        if column_specs is None:
            column_specs = self._spec
        else:
            self._spec = column_specs

        if not column_specs:
            raise ValueError('at least one column needs to be declared')

        if path is None:
            path = self.loader.get_file()
        if isinstance(path, str):
            path = Path(path)
        self.path = path

        self.empty_values += self.loader.empty_values

        col_names = []  # all row column headers, as in file
        col_index = []  # index of actually used columns
        keys = []
        field_names = []
        convfuncs = []

        cur_col_index = None  # for non-header input, defined by order in spec
        for spec_line in column_specs:
            # TODO: re-write as match statement?
            # FIXME: two-str-items format can be <col> <field> OR <field> <fun>
            # with no super easy way to distinguish them
            if not isinstance(spec_line, tuple):
                # no-header-simple-format
                spec_line = (spec_line, )

            if self.has_header is None:
                # detect header presence from first spec piece
                if len(spec_line) == 1:
                    self.has_header = False
                elif len(spec_line) == 2 and callable(spec_line[1]):
                    self.has_header = False
                else:
                    self.has_header = True

            if not self.has_header:
                spec_line = (self.NO_HEADER, *spec_line)

            colname, key, *convfunc = spec_line

            if len(convfunc) == 0:
                convfunc = None
            elif len(convfunc) == 1:
                convfunc = convfunc[0]
            else:
                raise ValueError(f'too many items in spec for {colname}/{key}')

            if colname is self.CALC_VALUE:
                if key is None:
                    raise RuntimeError('require key (field name) for for which'
                                       'to calculate a value')
                if convfunc is None:
                    raise ValueError('callable for value calculation missing')

                col_names.append(colname)
                col_index.append(cur_col_index)
            else:
                # current spec item is for a column in input
                if not self.has_header:
                    # add 0-based numerical index for column name
                    # these have to be counted here
                    if cur_col_index is None:
                        cur_col_index = 0
                    else:
                        cur_col_index += 1

                if key is None:
                    # ignore this column
                    continue

                if not self.has_header:
                    col_index.append(cur_col_index)

            col_names.append(colname)
            keys.append(key)

            if '.' in key:
                field_name, _, attr = key.partition('.')
                self.fk_attrs[field_name] = attr
            else:
                field_name = key

            field_names.append(field_name)

            if convfunc is None:
                pass
            elif isinstance(convfunc, str):
                convfunc_name = convfunc
                # getattr gives us a bound method:
                convfunc = getattr(loader, convfunc_name)
                if not callable(convfunc):
                    raise ValueError(
                        f'not the name of a {self.loader} method: '
                        f'{convfunc_name}'
                    )
            elif callable(convfunc):
                # Assume it's a function that takes the loader as
                # 1st arg.  We get this when the previoudsly
                # delclared method's identifier is passed directly
                # in the spec's declaration.
                convfunc = partial(convfunc, self.loader)
            else:
                raise ValueError(f'not a callable: {convfunc}')

            convfuncs.append(convfunc)

        self.col_names = col_names
        self.col_index = col_index
        self.keys = keys
        self.field_names = field_names
        self._convfuncs_raw = convfuncs

    def __len__(self):
        return len(self._spec)

    @property
    def fields(self):
        if self._fields is None:
            self._fields = tuple(
                (self.model._meta.get_field(i) for i in self.field_names)
            )
        return self._fields

    @property
    def convfuncs(self):
        if self._convfuncs is None:
            convfuncs = []
            for field, fn in zip(self.fields, self._convfuncs_raw):
                if fn is None and field.choices:
                    # automatically attach prep method for choice fields
                    convfuncs.append(
                        self.loader.get_choice_value_prep_function(field)
                    )
                else:
                    convfuncs.append(fn)
            self._convfuncs = tuple(convfuncs)

        return self._convfuncs

    def iterrows(self):
        """
        A generator for the records/rows of the input file

        This method must be implemented by inheriting classes.  The generator
        should yield a sequence of items for each record or row.
        """
        raise NotImplementedError

    def row_data(self, row):
        """
        Generate a row as tuples (field, func, value)

        Blank/empty values will be set to None here.  Extra items for
        calculated field values are added.

        :param list row: A list of str
        """
        it = zip(self.fields, self.convfuncs, self.col_names, self.col_index)
        for field, fn, col_name, col_i in it:
            if col_name is self.CALC_VALUE:
                # value will be calculated
                if fn is None:
                    raise ValueError(f'expect a callable but got {fn}')
                value = None
            else:
                value = row[col_i]
                if value in self.empty_values or value in field.empty_values:
                    value = None
            yield (field, fn, value)

    def row2dict(self, row):
        """ turn a row (of values) into a dict with field names as keys """
        return {field.name: val for field, _, val in self.row_data(row)}


class CSV_Spec(InputFileSpec):
    def __init__(self, *column_specs, sep='\t'):
        super().__init__(*column_specs)
        self.sep = sep

    def setup(self, *args, sep=None, **kwargs):
        super().setup(*args, **kwargs)
        if sep is not None:
            self.sep = sep

    def process_header(self, file):
        """
        consumes and checks a single line of columns headers

        Calling this will set up the column_index that is needed to iterate
        over the rows.

        Overwrite this method if your file has a more complex layout.  Make
        sure this method consumes all non-data rows at the beginning of the
        file.
        """
        head = file.readline().rstrip('\n').split(self.sep)
        col_pos = {colname: pos for pos, colname in enumerate(head)}

        column_index = []
        for col in self.col_names:
            if col is self.CALC_VALUE:
                column_index.append(col_pos)
            else:
                try:
                    pos = col_pos[col]
                except KeyError:
                    raise RuntimeError(
                        f'column not found: {col}, header: {head}'
                    )

                column_index.append(pos)
        self.col_index = column_index

    def iterrows(self):
        """
        An iterator over the csv file's rows

        :param pathlib.Path path: path to the data file
        """
        with self.path.open() as f:
            print(f'File opened: {f.name}')
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)

            if self.has_header:
                self.process_header(f)

            for line in f:
                yield line.rstrip('\n').split(self.sep)


class ExcelSpec(InputFileSpec):
    def __init__(self, *column_specs, sheet_name=None):
        super().__init__(*column_specs)
        self.sheet_name = sheet_name

    def get_dataframe(self):
        """ Return file as pandas.DataFrame """
        print(f'File opened: {self.path}')
        df = pandas.read_excel(
            str(self.path),
            sheet_name=self.sheet_name,
            # TODO: only read columns we need
            # usecols=...,
            na_values=self.empty_values,
            keep_default_na=False,
        )
        # turn NaNs (all the empty cells) into Nones
        # Else we'd have to tell load() to treat NaNs as empty
        df = df.where(pandas.notnull(df), None)
        return df

    def process_header(self, df):
        """
        Populate the column index
        """
        col_pos = {colname: pos for pos, colname in enumerate(df.columns)}

        column_index = []
        for col in self.col_names:
            if col is self.CALC_VALUE:
                column_index.append(col_pos)
            else:
                try:
                    pos = col_pos[col]
                except KeyError:
                    raise RuntimeError(
                        f'column not found: {col=} in {col_pos=}'
                    )

                column_index.append(pos)
        self.col_index = column_index

    def iterrows(self):
        """
        An iterator over the table's rows, yields Pandas.Series instances

        :param pathlib.Path path: path to the data file
        """
        df = self.get_dataframe()
        self.process_header(df)  # this call has to be made somewhere
        for _, row in df.iterrows():
            yield row


class SizedIterator:
    """
    a wrapper to attach a known length to an iterator

    Example usage:

    g = (do_something(i) for i in a_list)
    g = SizedIterator(g, len(a_list))
    len(g)

    """
    def __init__(self, obj, length):
        self._it = iter(obj)
        self._length = length

    def __iter__(self):
        return self._it

    def __next__(self):
        return next(self._it)

    def __len__(self):
        return self._length


def siter(obj, length=None):
    """
    Return a sized iterator

    Convenience function for using the SizedIterator.  If length is not given
    then len(obj) must work.  Compare to the one-argument form of the built-in
    iter() function
    """
    if length is None:
        length = len(obj)
    return SizedIterator(obj, length)


def atomic_dry(f):
    """
    Replacement for @atomic decorator for Manager methods

    Supports dry_run keyword arg and calls set_rollback appropriately and
    coordinates the db alias in case we have multiple databases.  This assumes
    that the decorated method has self.model available, as it is the case for
    managers, and that if write operations are used for other models then those
    must run on the same database connection.
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        dbalias = router.db_for_write(self.model)
        with transaction.atomic(using=dbalias):
            if 'dry_run' in kwargs:
                dry_run = kwargs['dry_run']
                if dry_run or 'dry_run' not in signature(f).parameters:
                    # consume dry_run kw if True as to avoid nested rollback
                    # but pass on a dry_run=False if wrapped function supports
                    # it, as to override any nested defaults saying otherwise
                    kwargs.pop('dry_run')
            else:
                dry_run = None
            retval = f(self, *args, **kwargs)
            if dry_run:
                transaction.set_rollback(True, dbalias)
            return retval
    return wrapper
