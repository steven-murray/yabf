"""Module defining data loaders for YAML files."""
import numpy as np
import os
import pickle

from .plugin import plugin_mount_factory


class LoadError(ValueError):
    pass


class DataLoader(metaclass=plugin_mount_factory()):
    def load(self, data):
        pass


class DictLoader(DataLoader):
    def load(self, data):
        if type(data) is not dict:
            raise LoadError()

        return data


class PickleLoader(DataLoader):
    def load(self, data):
        try:
            with open(data, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            raise
        except Exception:
            raise LoadError()


class npzLoader(DataLoader):
    def load(self, data):
        try:
            return dict(np.load(data))
        except FileNotFoundError:
            raise
        except Exception:
            raise LoadError()


class npyLoader(DataLoader):
    def load(self, data):
        try:
            return np.load(data)
        except FileNotFoundError:
            raise
        except Exception:
            raise LoadError()


class ValueLoader(DataLoader):
    def load(self, data):
        return data


# class HDF5Loader(DataLoader):
#     def load(self, data):
#         raise NotImplementedError()


class CompositeLoader(DataLoader):
    def __init__(self, loaders=None):
        self.loaders = loaders or DataLoader._plugins.values()

    def load(self, data):
        if not isinstance(data, str) or not os.path.exists(data):
            return data

        for loader in self.loaders:
            if loader is self.__class__:
                continue

            try:
                return loader().load(data)
            except LoadError:
                pass

        raise LoadError(
            f"None of the specified loaders were able to load the data: {data}"
        )
