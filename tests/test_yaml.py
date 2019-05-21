import os

import pytest
from yaml.scanner import ScannerError

from yabf.core import yaml


def _write(inner, outer, tmpdir):
    with open(os.path.join(tmpdir, "outer.yml"), 'w') as fl:
        fl.write(outer)
    with open(os.path.join(tmpdir, "inner.yml"), 'w') as fl:
        fl.write(inner)


def _load(tmpdir):
    with open(os.path.join(tmpdir, "outer.yml")) as fl:
        stuff = yaml.load(fl)
    return stuff


def test_include(tmpdir):
    outer = """
!include inner.yml
this: that    
    """

    inner = """
inner: inner    
    """

    _write(inner, outer, tmpdir)

    with pytest.raises(ScannerError):
        _load(tmpdir)

    outer = """
!include inner.yml
    """

    _write(inner, outer, tmpdir)
    stuff = _load(tmpdir)
    assert "inner" in stuff

    outer = """
params:
    - !include inner.yml
    - another
    """

    _write(inner, outer, tmpdir)
    stuff = _load(tmpdir)
    assert 'params' in stuff
    assert stuff['params'][0] == {"inner": "inner"}

    inner = '''
- inner    
    '''

    _write(inner, outer, tmpdir)
    stuff = _load(tmpdir)
    assert len(stuff['params']) == 2


def test_hard_include(tmpdir):
    outer = """
dct:
    a: another
    _incl_: !include_here inner.yml
    """

    inner = """
b: this    
    """

    _write(inner, outer, tmpdir)
    stuff = _load(tmpdir)
    assert stuff['dct'] == {'a': "another", "b": "this"}

    outer = """
- !include_here inner.yml
- this
- that
- the other    
    """

    inner = """
- those        
    """

    _write(inner, outer, tmpdir)
    stuff = _load(tmpdir)
    assert stuff == ['those', 'this', 'that', 'the other']
