#!/usr/bin/env python
"""Utility library for "chunked" custom search commands."""

from collections import OrderedDict
import csv
import logging
import sys
import time
import traceback

if sys.version_info >= (3, 0):
    from io import StringIO
else:
    from cStringIO import StringIO

from cexc import setup_logging
from util.chunk_util import read_chunk, write_chunk

logger = setup_logging.get_logger()
messages = logger.getChild('messages')


class CommandType(object):
    """Chunked Command Protocol command types"""

    EVENTS = 'events'
    REPORTING = 'reporting'
    STREAMING = 'streaming'
    STATEFUL = 'stateful'


def get_logger(name=None):
    """Returns a logger for internal messages."""
    if name:
        return logger.getChild(name)
    else:
        return logger


def get_messages_logger():
    """Returns a logger for user-visible messages."""
    return messages


def log_traceback():
    """Logs a traceback. Useful in Exception handlers."""
    logger.error(traceback.format_exc())


def abort(msg):
    """Helper method to abort gracefully with a user-visible message.

    Do NOT use this method from within a running
    BaseChunkHandler. Instead, raise an Exception or RuntimeError.

    Invoke this function to gracefully exit a custom search command
    before a BaseChunkHandler object has been created and run. You may
    use this, for instance, if there is an exception during an import
    in your __main__ module.
    """
    AbortHandler.abort(msg)


class BaseChunkHandler(object):
    """Base class for custom search commands using the "chunked" protocol.

    This is a low-level implementation. You are strongly encouraged to
    use the Splunk Python SDK instead.

    To write an external search command, extend this class, override
    the handler() method, and invoke its run() method, e.g.:

        class Handler(BaseChunkHandler):
            def handler(self, metadata, data):
                ...
        if __name__ == "__main__":
            Handler().run()

    run() will read a chunk from stdin, call handler() with the
    metadata and data payloads, and write a chunk containing
    handler()'s return value. It will continue doing this in a loop
    until EOF is read.

    Parameters
    ----------
    handler_data : DATA_DICT | DATA_CSVROW | DATA_RAW
        Specifies how/whether data payload should be parsed.
        Defaults to DATA_DICT.

    in_file, out_file, err_file : file
        Files to use for input, output, and errors, respectively.
        Defaults to sys.stdin.buffer, sys.stdout.buffer, sys.stderr.
        N.B.: in_file must be a byte stream, where in_file.read(num) reads
        num bytes (and not characters). This is because `body_len` in chunk
        protocol specifies body length in bytes.

    Attributes
    ----------
    getinfo : dict, class attribute
        Metadata from the getinfo exchange. Set when
        action:getinfo is observed in _read_chunk().

    """

    (
        DATA_DICT,  # parse data payload with csv.DictReader
        DATA_CSVROW,  # parse data payload with csv.reader
        DATA_RAW,  # don't parse data payload
    ) = list(range(3))

    def __init__(
        self,
        handler_data=None,
        in_file=sys.stdin.buffer,
        out_file=sys.stdout.buffer,
        err_file=sys.stderr,
    ):
        if handler_data:
            self.handler_data = handler_data
        else:
            self.handler_data = self.DATA_DICT
        self.in_file = in_file
        self.out_file = out_file
        self.err_file = err_file
        self.getinfo = {}

        # Unmangle line-endings in Windows.

        # N.B. : Windows converts \n to \r such that transport headers do not
        # get received correctly by the CEXC protocol. However, this is really
        # only needed when the IO is actually an object with a file descriptor.
        # Python 2 docs note that file-like objects that don't have real file
        # descriptors should *not* implement a fileno method:

        if sys.platform == "win32":
            import os, msvcrt  # pylint: disable=import-error

            for file_like_object in [self.in_file, self.out_file, self.err_file]:
                fileno = getattr(file_like_object, 'fileno', None)
                if fileno is not None:
                    if callable(fileno):
                        try:
                            msvcrt.setmode(
                                file_like_object.fileno(), os.O_BINARY
                            )  # pylint: disable=E1103 ; the Windows version of os has O_BINARY
                        except ValueError:
                            # This can be safely skipped, as it is raised
                            # from pytest which incorreclty implements a fileno
                            pass

        # Logger instance for user-visible messages.
        self.messages_logger = get_messages_logger()
        self.messages_handler = logging.handlers.BufferingHandler(100000)
        self.messages_logger.addHandler(self.messages_handler)

        # Variables to track time spent in different chunk handling
        # states.
        self._read_time = 0.0
        self._handle_time = 0.0
        self._write_time = 0.0

        self.controller_options = None
        self.controller = None
        self.watchdog = None
        self.partial_fit = None

    def run(self):
        """Handle chunks in a loop until EOF is read.

        If an exception is raised during chunk handling, a chunk
        indicating the error will be written and the process will exit.
        """
        try:
            while self._handle_chunk():
                pass
        except Exception as e:
            if isinstance(e, RuntimeError):
                error_message = str(e)
            else:
                error_message = '(%s) %s' % (type(e).__name__, e)
            self.die(error_message)

    def handler(self, metadata, body):
        """Default chunk handler, returns empty metadata and data payloads."""
        return ({}, [])

    def die(self, message, log_traceback=True):
        """Logs a message, writes a user-visible error, and exits."""

        logger.error(message)
        if log_traceback:
            logger.error(traceback.format_exc())

        metadata = {'finished': True, 'error': message}

        # Insert inspector messages from messages_logger.
        messages = self._pop_messages()
        # Convert non-DEBUG messages to ERROR so the user can see them...
        messages = [['ERROR', y] for x, y in messages if x != 'DEBUG']

        if len(messages) > 0:
            metadata.setdefault('inspector', {}).setdefault('messages', []).extend(messages)

        # Sort the keys in reverse order! 'inspector' must come before 'error'.
        metadata = OrderedDict([(k, metadata[k]) for k in sorted(metadata, reverse=True)])

        write_chunk(self.out_file, metadata, '')
        sys.exit(1)

    def _handle_chunk(self):
        """Handle (read, process, write) a chunk."""
        with Timer() as t:
            ret = read_chunk(self.in_file)
            if not ret:
                return False  # EOF

            metadata, body = ret

            if self.handler_data == self.DATA_DICT:
                assert isinstance(body, str)
                body = list(csv.DictReader(StringIO(body)))
            elif self.handler_data == self.DATA_CSVROW:
                assert isinstance(body, str)
                body = list(csv.reader(StringIO(body)))
            elif self.handler_data == self.DATA_RAW:
                pass

            # Cache a copy of the getinfo metadata.
            if metadata.get('action', None) == 'getinfo':
                self.getinfo = dict(metadata)

        self._read_time += t.interval

        with Timer() as t:
            # Invoke handler. Hopefully someone overloaded it!
            ret = self.handler(metadata, body)

            if isinstance(ret, dict):
                metadata, body = ret, None
            else:
                try:
                    metadata, body = ret
                except:
                    raise TypeError(
                        "Handler must return (metadata, body), got: %.128s" % repr(ret)
                    )

            # Insert inspector messages from messages_logger.
            messages = self._pop_messages()
            if len(messages) > 0:
                metadata.setdefault('inspector', {}).setdefault('messages', []).extend(messages)

        self._handle_time += t.interval

        with Timer() as t:
            if body is not None and len(body) > 0:
                sio = StringIO()

                if self.handler_data == self.DATA_DICT:
                    assert hasattr(body, '__iter__')

                    keys = set()
                    for r in body:
                        keys.update(list(r.keys()))

                    writer = csv.DictWriter(sio, fieldnames=list(keys))
                    writer.writeheader()

                    for r in body:
                        writer.writerow(r)
                    body = sio.getvalue()

                elif self.handler_data == self.DATA_CSVROW:
                    writer = csv.writer(sio)
                    for r in body:
                        writer.writerow(r)
                    body = sio.getvalue()
                elif self.handler_data == self.DATA_RAW:
                    pass

                assert isinstance(body, str)

            else:
                body = ''

            write_chunk(self.out_file, metadata, body)

        self._write_time += t.interval

        return True

    def _pop_messages(self):
        """Drain logging.MemoryHandler holding user-visible messages."""
        messages = []
        for r in self.messages_handler.buffer:
            # Map message levels to Splunk equivalents.
            level = {
                'DEBUG': 'DEBUG',
                'INFO': 'INFO',
                'WARNING': 'WARN',
                'ERROR': 'ERROR',
                'CRITICAL': 'ERROR',
            }[r.levelname]
            messages.append([level, r.message])

        self.messages_handler.flush()
        return messages


class AbortHandler(BaseChunkHandler):
    def __init__(self, msg):
        self.msg = msg
        super(AbortHandler, self).__init__()

    def handler(self, metadata, body):
        raise RuntimeError(self.msg)

    @classmethod
    def abort(cls, msg):
        cls(msg).run()
        sys.exit(1)


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
