import contextlib
import re
import sys
from io import StringIO
from typing import TextIO


def filter_io(file_in: TextIO, filter_regex: str, file_out: TextIO) -> None:
    """
    Filter out lines in a TextIO object and writing lines that pass the filter into another io

    :param file_in: TextIO object to be filtered, respects cursor location
    :param filter_regex: regex string of items to be filtered out
    :param file_out: output TextIO to write to
    :return: None
    """
    matcher = re.compile(filter_regex)
    for line in file_in:
        if matcher.search(line) is None:
            file_out.write(line)


class filter_stderr(contextlib.AbstractContextManager):
    """
    Context manager for buffering stderr and suppressing any lines that match a regex.
    Note that this context manager will buffer all output to stderr until it exits, at which
    point it will output all non-matching lines in order.
    """

    def __init__(self, regex_str: str):
        self.io_stream = StringIO()
        self.regex_str = regex_str
        self.redirect_stderr = contextlib.redirect_stderr(self.io_stream)

    def __enter__(self):
        # This line initiates the redirection of stderr to self.io_stream
        self.redirect_stderr.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # After the redirect_stderr context manager exits, stderr will be restored
        ret_val = self.redirect_stderr.__exit__(exc_type, exc_val, exc_tb)
        # Go through the io_stream which contains redirected outputs and put them
        # back into sys.stderr if they do not match the regex.
        self.io_stream.seek(0)
        filter_io(self.io_stream, self.regex_str, sys.stderr)
        return ret_val
