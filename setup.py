#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import find_packages, setup


project_dir = Path(__file__).parent

# Import the README and use it as the long-description.
with open(project_dir.joinpath('README.md')) as f:
    LONG_DESCRIPTION = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
info = {}
with open(project_dir.joinpath('madminer', '__info__.py')) as f:
    exec(f.read(), info)


# Package meta-data.
NAME = 'madminer'
DESCRIPTION = 'Mining gold from MadGraph to improve limit setting in particle physics.'
URL = 'https://github.com/diana-hep/madminer'
EMAIL = 'johann.brehmer@nyu.edu'
REQUIRES_PYTHON = '>=3.6, <4'
AUTHORS = info['__authors__']
VERSION = info['__version__']
REQUIRED = [
    "h5py",
    "matplotlib>=2.0.0",
    "numpy>=1.13.0",
    "particle>=0.15.1",
    "scipy>=1.0.0",
    "torch>=1.0.0",
    "uproot3>=3.14.1",
    "vector>=0.8.4",
]

EXTRAS_DOCS = [
    "myst-parser",
    "numpydoc",
    "sphinx>=1.4",
    "sphinx_rtd_theme",
]

EXTRAS_TEST = [
    "pytest",
]

EXTRAS_EXAMPLES = [
    "bqplot",
    "pandas",
]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHORS,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    extras_require={
        "docs": EXTRAS_DOCS,
        "test": EXTRAS_TEST,
        "examples": EXTRAS_EXAMPLES,
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
