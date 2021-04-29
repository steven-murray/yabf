"""
Overload of yaml which supports an !include directive.

Swiped from https://gist.github.com/joshbode/569627ced3076931b02f

"""
import yaml
from pathlib import Path

from .io import DataLoader


def register_data_loader_tags():
    for loader in DataLoader._plugins.values():
        ld = loader()

        def fnc(loader, node):
            pth = Path(node.value)
            new_path = Path(loader._root) / pth if not pth.exists() else pth
            return ld.load(new_path)

        yaml.add_constructor(f"!{ld.tag}", fnc, Loader=yaml.FullLoader)
        yaml.add_constructor(f"!{ld.tag}", fnc, Loader=yaml.Loader)
