from datetime import datetime
from itertools import zip_longest
from threading import Timer
from string import Formatter
import sys


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

    After the timer has stopped, even after calling finish() or stop() the
    progress printing can be resumed by calling update() with a different state
    than the last one.
    """
    DEFAULT_TEMPLATE = '{progress}'
    DEFAULT_INTERVAL = 1.0  # seconds

    def __init__(
            self,
            template=DEFAULT_TEMPLATE,
            interval=DEFAULT_INTERVAL,
            output_file=sys.stdout,
            show_rate=True,
    ):
        self.template, self.template_var = self._init_template(template)
        self.interval = interval
        self.output_file = output_file
        self.show_rate = show_rate
        self.to_terminal = output_file.isatty()

        self._reset_state()

    def _reset_state(self):
        """ reset the variable state """
        self.current = None
        self.last = None
        self.at_previous_ring = None
        self.timer = None
        self.timer_running = False
        self.time_zero = datetime.now()

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

    def _start_timer(self):
        self.timer = Timer(self.interval, self._ring)
        self.timer.start()
        self.timer_running = True

    def stop(self):
        """ Stop the timer """
        try:
            self.timer.cancel()
        except Exception:
            pass
        self.timer_running = False

    def update(self, current):
        """
        Update current state

        Print (for first time) turn on timer for non-boring progress
        """
        self.last = self.current
        self.current = current

        if not self.timer_running and current != self.last:
            self.print_progress()  # print on first update
            # turn on
            self._start_timer()

    def inc(self, step=1):
        """
        Assume we're progressing by counting integers and increment
        """
        count = self.current
        if count is None:
            count = step
        else:
            count += step
        self.update(count)

    def finish(self):
        """ Stop the timer but print a final result """
        self.stop()
        total_seconds = (datetime.now() - self.time_zero).total_seconds()
        avg_txt = (f'(total: {total_seconds:.1f}s '
                   f'avg: {self.current / total_seconds:.1f}/s)')
        self.print_progress(avg_txt=avg_txt, end='\n')  # print with totals/avg
        self._reset_state()

    def _ring(self):
        """ Print and restart timer """
        if self.current is None:
            return

        if self.current == self.last:
            return

        self.print_progress()
        self.last = self.current  # ensure we'll turn off without updates
        self.at_previous_ring = self.current
        self._start_timer()

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
                txt += f' ({rate:.0f}/s)'

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
