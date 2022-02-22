from datetime import datetime
from itertools import zip_longest
from threading import Lock, Timer
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
            end=None,
    ):
        self.template, self.template_var = self._init_template(template)
        self.interval = interval
        self.output_file = output_file
        self.show_rate = show_rate
        self.to_terminal = output_file.isatty()
        self.end = end
        self.it = None
        self.current = None
        self.timer_lock = Lock()
        self.timer = None

    def __call__(self, it):
        self.it = it
        self._reset_state()
        for elem in it:
            yield elem
            self.inc()
        self.it = None
        self.finish()

    def _reset_state(self):
        """
        reset the variable state

        Should be called once before anything interesting is done
        """
        self.current = 0
        self.last = None
        self.at_previous_ring = None
        self.timer_lock.acquire
        self.ring_time = None
        self.time_zero = datetime.now()
        if self.end is None:
            if self.it is None:
                self._end = None
            else:
                if isinstance(self.it, list):
                    self._end = len(self.it)
                else:
                    self._end = None
        else:
            self._end = self.end

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
        if self.current is None:
            self._reset_state()

        if self.current == 0:
            self._set_timer()

        self.last = self.current
        self.current += step

    def _set_timer(self, reset=False):
        """
        Set or reset the timer

        param: reset -- True if we were called via _ring(), assumed to be False
                        for the initial call via inc()
        """
        # this can be called (1) by inc() of main thread and (2) by _ring() of
        # timer thread.  In either case, we (re-)set the timer only if we get
        # the non-blocking lock and think that no timer is currently running.
        # If the lock it taken, assume the other thread will reset it.
        if self.timer_lock.acquire(blocking=False):
            if reset or self.timer is None:
                self.timer = Timer(self.interval, self._ring)
                self.timer.start()
            else:
                # never start a second timer, we shouldn't get here by normal
                # use, but someone might call _set_timer() manually or so
                pass
            self.timer_lock.release()

    def finish(self):
        """ Stop the timer and print a final result """
        if self.current is None:
            # zero-length iterator?
            # populate variables needed for printing
            self._reset_state()
        total_seconds = (datetime.now() - self.time_zero).total_seconds()

        self.timer_lock.acquire()
        # get lock so we don't cancel while also trying to re-set the timer in
        # _set_timer()
        try:
            self.timer.cancel()
        except Exception:
            pass
        self.timer_lock.release()

        avg_txt = (f'(total: {total_seconds:.1f}s '
                   f'avg: {self.current / total_seconds:.1f}/s)')
        self.print_progress(avg_txt=avg_txt, end='\n')  # print with totals/avg
        self._reset_state()

    def _ring(self):
        """ Print and restart timer """
        if self.current == 0:
            # finish() called
            return

        now = datetime.now()
        self.ring_time = now

        self.print_progress()

        if self.current == self.at_previous_ring:
            # maybe we just iterate very slowly relative to the timer interval,
            # but probably some exception occurred in the main thread; have to
            # stop the timer or we'll get an infinite loop
            return

        self.at_previous_ring = self.current
        self._set_timer(reset=True)

    def estimate(self):
        """
        get current percentage and estimated finish time

        Returns None if we havn't made any progress yet or we don't know the
        length of the iterator.
        """
        if self._end in [None, 0] or self.current == 0:
            return None

        frac = self.current / self._end
        remain = (self.ring_time - self.time_zero).total_seconds() * (self._end / self.current - 1)  # noqa:E501
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
