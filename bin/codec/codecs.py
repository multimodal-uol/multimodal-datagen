#!/usr/bin/env python

import base64
import importlib
from io import BytesIO

import pandas as pd

CODECS = {
    ("builtins", "object"): "NoopCodec",
    ("builtins", "slice"): "SliceCodec",
    ("builtins", "set"): "SetCodec",
    ("builtins", "type"): "TypeCodec",
    ("builtins", "bytes"): "BytesCodec",
    ("numpy", "ndarray"): "NDArrayCodec",
    ("numpy", "int8"): "NDArrayWrapperCodec",
    ("numpy", "int16"): "NDArrayWrapperCodec",
    ("numpy", "int32"): "NDArrayWrapperCodec",
    ("numpy", "int64"): "NDArrayWrapperCodec",
    ("numpy", "uint8"): "NDArrayWrapperCodec",
    ("numpy", "uint16"): "NDArrayWrapperCodec",
    ("numpy", "uint32"): "NDArrayWrapperCodec",
    ("numpy", "uint64"): "NDArrayWrapperCodec",
    ("numpy", "float16"): "NDArrayWrapperCodec",
    ("numpy", "float32"): "NDArrayWrapperCodec",
    ("numpy", "float64"): "NDArrayWrapperCodec",
    ("numpy", "float128"): "NDArrayWrapperCodec",
    ("numpy", "complex64"): "NDArrayWrapperCodec",
    ("numpy", "complex128"): "NDArrayWrapperCodec",
    ("numpy", "complex256"): "NDArrayWrapperCodec",
    ("numpy", "dtype"): "DTypeCodec",
    ("mtrand", "RandomState"): "mtrandCodec",
    ("scipy.sparse.csr", "csr_matrix"): "SparseMatrixCodec",
    ("sklearn.dummy", "DummyClassifier"): "SimpleObjectCodec",
    ("sklearn.dummy", "DummyRegressor"): "SimpleObjectCodec",
    ("pandas.core.frame", "DataFrame"): "SimpleObjectCodec",
    ("pandas.core.index", "Index"): "IndexCodec",
    ("pandas.core.indexes.base", "Index"): "IndexCodec",
    ("pandas.core.indexes.range", "RangeIndex"): "IndexCodec",
    ("pandas.core.index", "Int64Index"): "IndexCodec",
    ("pandas.core.internals.managers", "BlockManager"): "BlockManagerCodec",
}


class BaseCodec(object):
    @classmethod
    def encode(cls, obj):
        raise NotImplementedError("Encoder not implemented")

    @classmethod
    def decode(cls, obj):
        raise NotImplementedError("Decoder not implemented")


class NoopCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        return {"__mlspl_type": [type(obj).__module__, type(obj).__name__]}

    @classmethod
    def decode(cls, obj):
        module_name, name = obj["__mlspl_type"]
        module = importlib.import_module(module_name)
        class_ref = getattr(module, name)

        new_obj = class_ref.__new__(class_ref)

        return new_obj


class BytesCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "str": obj.decode("utf8"),
        }

    @classmethod
    def decode(cls, obj):
        return obj["str"].encode("utf")


class SliceCodec(BaseCodec):
    whitelist = [k for k, v in list(CODECS.items()) if v == "SliceCodec"]

    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__
        assert (module, name) in cls.whitelist

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "slice": obj.__reduce__()[1],
        }

    @classmethod
    def decode(cls, obj):
        module_name, name = obj["__mlspl_type"]
        assert (module_name, name) in cls.whitelist

        module = importlib.import_module(module_name)
        class_ref = getattr(module, name)

        new_obj = class_ref(*obj["slice"])

        return new_obj


class SimpleObjectCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__
        return {"__mlspl_type": [module, name], "dict": obj.__dict__}

    @classmethod
    def decode(cls, obj):
        module_name, name = obj["__mlspl_type"]

        module = importlib.import_module(module_name)
        class_ref = getattr(module, name)
        new_obj = class_ref.__new__(class_ref)
        new_obj.__dict__ = obj["dict"]
        for key in new_obj.__dict__:
            if isinstance(new_obj.__dict__[key], list) or isinstance(
                new_obj.__dict__[key], pd.Index
            ):
                new_obj.__dict__[key] = [
                    item if isinstance(item, str) else item for item in new_obj.__dict__[key]
                ]
            elif isinstance(new_obj.__dict__[key], str):
                new_obj.__dict__[key] = new_obj.__dict__[key]
        return new_obj


class IndexCodec(BaseCodec):
    whitelist = [k for k, v in list(CODECS.items()) if v == "IndexCodec"]

    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__
        assert (module, name) in cls.whitelist

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "init_args": obj.__reduce__()[1][1],
        }

    @classmethod
    def decode(cls, obj):
        module_name, name = obj["__mlspl_type"]
        assert (module_name, name) in cls.whitelist

        module = importlib.import_module(module_name)
        class_ref = getattr(module, name)

        new_obj = class_ref(**obj["init_args"])

        return new_obj  # pandas.core.index.Index(**obj['init_args'])


class DTypeCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "descr": obj.descr if obj.names is not None else obj.str,
        }

    @classmethod
    def decode(cls, obj):
        import numpy as np

        return np.dtype(obj["descr"])


class NDArrayWrapperCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        import numpy as np

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "ndarray": np.array([obj]),
        }

    @classmethod
    def decode(cls, obj):
        return obj["ndarray"][0]


class NDArrayCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        import numpy as np

        assert type(obj) == np.ndarray

        if obj.dtype.hasobject:
            try:
                obj = obj.astype("U")
            except Exception:
                raise ValueError("Cannot encode numpy.ndarray with objects")

        bio = BytesIO()
        np.save(bio, obj, allow_pickle=False)

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "npy": base64.b64encode(bio.getvalue()).decode("utf8"),
        }

    @classmethod
    def decode(cls, obj):
        import numpy as np

        as_string = obj["npy"].encode("utf8")
        b64decoded = base64.b64decode(as_string)
        bio = BytesIO(b64decoded)
        return np.load(bio, allow_pickle=False)


class TreeCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        import sklearn.tree

        assert type(obj) == sklearn.tree._tree.Tree

        init_args = obj.__reduce__()[1]
        state = obj.__getstate__()

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "init_args": init_args,
            "state": state,
        }

    @classmethod
    def decode(cls, obj):
        import sklearn.tree

        init_args = obj["init_args"]

        state = obj["state"]

        # Add max_depth for backwards compatibility with PSC 1.2
        # Previous version did not set the max_depth in the state when calling __getstate__
        # https://github.com/scikit-learn/scikit-learn/blob/51a765acfa4c5d1ec05fc4b406968ad233c75162/sklearn/tree/_tree.pyx#L615

        # and has been added in sklearn 0.18 to be used in both __getstate__ and __setstate__
        # https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a805efbe4bb06516670a9b8c690992bd7/sklearn/tree/_tree.pyx#L649

        # Older models will not have the max_depth in their stored state, such that a key error is raised.
        # the max_depth is only used in the decision path method, which we don't currently use
        # and is used to init an np array of zeros in version 0.18:
        # https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a805efbe4bb06516670a9b8c690992bd7/sklearn/tree/_tree.pyx#L926
        # https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a805efbe4bb06516670a9b8c690992bd7/sklearn/tree/_tree.pyx#L991
        state["max_depth"] = state.get("max_depth", 0)

        t = sklearn.tree._tree.Tree(*init_args)

        t.__setstate__(state)

        return t


class BlockManagerCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        from pandas.core.internals import BlockManager, SingleBlockManager

        assert type(obj) == BlockManager or type(obj) == SingleBlockManager

        state = obj.__getstate__()

        return {"__mlspl_type": [type(obj).__module__, type(obj).__name__], "state": state}

    @classmethod
    def decode(cls, obj):
        from pandas.core.internals import BlockManager, SingleBlockManager

        state = obj["state"]

        if obj['__mlspl_type'][1] == 'BlockManager':
            t = BlockManager.__new__(BlockManager)
        elif obj['__mlspl_type'][1] == 'SingleBlockManager':
            t = SingleBlockManager.__new__(SingleBlockManager)
        t.__setstate__(state)

        return t


class SetCodec(BaseCodec):
    whitelist = [k for k, v in list(CODECS.items()) if v == "SetCodec"]

    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__
        assert (module, name) in cls.whitelist

        return {"__mlspl_type": [type(obj).__module__, type(obj).__name__], "set": list(obj)}

    @classmethod
    def decode(cls, obj):
        module_name, name = obj["__mlspl_type"]
        assert (module_name, name) in cls.whitelist

        return set(obj["set"])


class TypeCodec(BaseCodec):
    whitelist = [k for k, v in list(CODECS.items()) if v == "TypeCodec"]

    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__
        assert (module, name) in cls.whitelist

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "type": [obj.__module__, obj.__name__],
        }

    @classmethod
    def decode(cls, obj):
        module_name, name = obj["__mlspl_type"]
        assert (module_name, name) in cls.whitelist
        assert (obj["type"][0], obj["type"][1]) in CODECS

        module = importlib.import_module(obj["type"][0])

        return getattr(module, obj["type"][1])


class mtrandCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        import numpy as np

        assert type(obj) == np.random.mtrand.RandomState

        init_args = obj.__reduce__()[1]
        state = obj.__getstate__()

        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "init_args": init_args,
            "state": state,
        }

    @classmethod
    def decode(cls, obj):
        from numpy.random.mtrand import RandomState

        init_args = obj["init_args"]
        state = obj["state"]

        t = RandomState(*init_args)
        t.__setstate__(state)

        return t


class SparseMatrixCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        import numpy as np
        from scipy import sparse

        assert type(obj) == sparse.csr.csr_matrix

        bio = BytesIO()
        np.savez(bio, data=obj.data, indices=obj.indices, indptr=obj.indptr, shape=obj.shape)
        return {
            "__mlspl_type": [type(obj).__module__, type(obj).__name__],
            "sparse_npy": base64.b64encode(bio.getvalue()),
        }

    @classmethod
    def decode(cls, obj):
        import numpy as np
        from scipy.sparse.csr import csr_matrix

        bio = BytesIO(base64.b64decode(obj["sparse_npy"]))
        loader = np.load(bio)
        return csr_matrix(
            (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
        )


class SeriesCodec(BaseCodec):
    whitelist = [k for k, v in CODECS.items() if v == 'SeriesCodec']

    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__
        assert (module, name) in cls.whitelist
        state = obj.__getstate__()
        return {'__mlspl_type': [type(obj).__module__, type(obj).__name__], 'state': state}

    @classmethod
    def decode(cls, obj):
        import pandas.core.series

        module_name, name = obj['__mlspl_type']
        assert (module_name, name) in cls.whitelist

        state = obj['state']
        t = pandas.core.series.Series.__new__(pandas.core.series.Series)
        t.__setstate__(state)
        return t
