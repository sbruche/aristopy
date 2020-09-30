#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages

def read_file(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()
setup(
    name='aristopy',
    version='0.9.2',
    # metadata to display on PyPI
    author="Stefan Bruche",
    author_email='stefan.bruche@tu-berlin.de',
    url='https://github.com/sbruche/aristopy',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    description="Framework for the optimization of energy systems",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    keywords=['energy systems', 'optimization', 'pyomo'],
    project_urls={'Documentation': 'https://aristopy.readthedocs.io/en/latest/index.html'},
    install_requires=['pandas>=0.19.2',
                      'numpy>=1.11.3',
                      'pyomo>=5.6.9',
                      'tsam>=1.1.0',
                      'xlrd>=1.0.0',
                      'openpyxl',
                      'matplotlib'],
    license="MIT license",
    packages=find_packages(include=['aristopy', 'aristopy.*']),
    setup_requires=['setuptools'],
    # allows installation with additional packages: $ pip install -e .[dev]
    extras_require={'dev': ['pytest',
                            'sphinx', 'sphinx-rtd-theme>=0.4.3',
                            'nbsphinx',
                            'ipython',
                            'twine']}
)

