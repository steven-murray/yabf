"""
Overload of yaml which supports an !include directive.

Swiped from https://gist.github.com/joshbode/569627ced3076931b02f

"""
import logging
import os.path
import yaml
from yaml import *  # noqa

log = logging.getLogger(__name__)


class ExtLoaderMeta(type):
    def __new__(mcs, __name__, __bases__, __dict__):
        """Add include constructor to class."""
        # register the include constructor on the class
        cls = super().__new__(mcs, __name__, __bases__, __dict__)
        cls.add_constructor("!include", cls.construct_include)
        cls.add_constructor("!include_here", cls.construct_include_here)

        return cls


class ExtLoader(yaml.FullLoader, metaclass=ExtLoaderMeta):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node))
        )
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r") as f:
            if extension in ("yaml", "yml"):
                return yaml.load(f, ExtLoader)
            else:
                return "".join(f.readlines())

    def construct_include_here(self, node):
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node))
        )
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r") as f:
            if extension in ("yaml", "yml"):
                out = yaml.load(f, ExtLoader)
                if isinstance(out, list):
                    out.append("__del__")
                elif isinstance(out, dict):
                    out["__del__"] = True

                return out
            else:
                return "".join(f.readlines())


def _move_up(obj, parent=None, indx=None):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            _move_up(v, obj, k)

            if "__del__" in obj and k != "__del__":
                parent[k] = v

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _move_up(item, obj, i)

            if "__del__" in obj and item != "__del__":
                parent.insert(indx + i + 1, item)

    if hasattr(obj, "__getitem__") and "__del__" in obj and obj != "__del__":
        del parent[indx]


# Set ExtLoader as default.
def load(*args, **kwargs):
    out = yaml.load(*args, Loader=ExtLoader, **kwargs)
    _move_up(out)
    return out
