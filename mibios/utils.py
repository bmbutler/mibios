"""
Utilities module
"""
import logging


class PrintLikeLogging(logging.LoggerAdapter):
    """
    Adapter to log just like print
    """
    def log(self, level, *msg, sep=' ',
            **kwargs):
        """
        Adapt to log just like print

        Except end, file, and flush keyword args are not to be used
        """
        super().log(level, sep.join([str(i) for i in msg]), **kwargs)


def getLogger(name):
    """
    Wrapper around logging.getLogger()
    """
    return PrintLikeLogging(logging.getLogger(name), {})
