from threading import Timer
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

    After the timer has stopped, even after calling finishe() or stop() the
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
    ):
        self.template = template
        self.interval = interval
        self.output_file = output_file
        self.to_terminal = output_file.isatty()
        self.current = None
        self.last = None
        self.timer = None
        self.timer_running = False

    def start_timer(self):
        self.timer = Timer(self.interval, self.ring)
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
            self.start_timer()

    def finish(self):
        """ Stop the timer but print a final result """
        self.stop()
        self.print_progress(end='\n')  # print a last time

    def ring(self):
        """ Print and restart timer """
        if self.current is None:
            return

        if self.current == self.last:
            return

        self.print_progress()
        self.last = self.current  # ensure we'll turn off without updates
        self.start_timer()

    def print_progress(self, end=''):
        """ Do the progress printing """
        txt = self.template.format(progress=self.current)
        if self.to_terminal:
            txt = '\r' + txt

        print(txt, end=end, flush=True, file=self.output_file)
