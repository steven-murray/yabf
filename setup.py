#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=6.0",
    "numpy>=1",
    "scipy>=1",
    "pyyaml>=5",
    "cached_property",
    "frozendict",
]

setup_requirements = [
    #    'pytest-runner',
]

test_requirements = ["pytest"]

setup(
    author="Steven Murray",
    author_email="steven.g.murray@asu.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Scientists",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Yet Another Bayesian Framework",
    entry_points={"console_scripts": ["yabf=yabf.cli:main"]},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="yabf",
    name="yabf",
    packages=find_packages(include=["yabf", "yabf.core"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/steven-murray/yabf",
    version="0.0.1",
    zip_safe=False,
)
