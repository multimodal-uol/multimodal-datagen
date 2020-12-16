#!/usr/bin/env python

import json
from distutils.version import StrictVersion

import numpy as np

from . import codecs_manager

NUMPY_VERSION = "1.10.0"


class MLSPLEncoder(json.JSONEncoder):
    def default(
        self, obj
    ):  # pylint: disable=E0202 ; pylint doesn't like overriding default for some reason
        codec = codecs_manager.get_codec_table().get(
            (type(obj).__module__, type(obj).__name__), None
        )
        if codec is not None:
            return codec.encode(obj)

        try:
            return json.JSONEncoder.default(self, obj)
        except Exception:
            raise TypeError(
                "Not JSON serializable: %s.%s" % (type(obj).__module__, type(obj).__name__)
            )


class MLSPLDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(MLSPLDecoder, self).__init__(*args, object_hook=self._object_hook, **kwargs)

    def _object_hook(self, obj):
        if isinstance(obj, dict) and "__mlspl_type" in obj:
            module_name, name = obj["__mlspl_type"]
            codec = codecs_manager.get_codec_table().get((module_name, name), None)
            if codec:
                return codec.decode(obj)
            raise ValueError('No codec for record of type "%s.%s"' % (module_name, name))
        return obj


def encode(obj):
    if StrictVersion(np.version.version) >= StrictVersion(NUMPY_VERSION):
        return MLSPLEncoder().encode(obj)
    else:
        raise RuntimeError(
            "Python for Scientific Computing version 1.1 or later is required to save models."
        )


def decode(payload):
    if StrictVersion(np.version.version) >= StrictVersion(NUMPY_VERSION):
        return MLSPLDecoder().decode(payload)
    else:
        raise RuntimeError(
            "Python for Scientific Computing version 1.1 or later is required to load models."
        )
