import contextlib
import sys

from io import StringIO


class FilteredStdout:
    def __init__(self, filter_text="are not expected by XFormersAttnProcessor and will be ignored"):
        self.original_stdout = sys.stdout
        self.filter_text = filter_text
        self.buffer = StringIO()

    def write(self, text):
        # Only write to the original stdout if the text doesn't contain our filter
        if self.filter_text not in text:
            self.original_stdout.write(text)
        # Store everything in our buffer
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def getvalue(self):
        return self.buffer.getvalue()


# Context manager to selectively filter stdout
@contextlib.contextmanager
def filter_stdout(filter_text="are not expected by XFormersAttnProcessor and will be ignored"):
    filtered_stdout = FilteredStdout(filter_text)
    original_stdout = sys.stdout
    sys.stdout = filtered_stdout

    try:
        yield filtered_stdout
    finally:
        sys.stdout = original_stdout
