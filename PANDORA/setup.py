# -*- coding: utf-8 -*-
import os
from pathlib import Path
from os.path import exists
import json

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit PANDORA/__version__.py
version = {}
with open(os.path.join(here, 'PANDORA', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='CSB-PANDORA',
    version=version['__version__'],
    description='Model peptide-MHC complexes using anchor distance restrains in MODELLER',
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author='Dario Marzella, Farzaneh Parizi, Li Xue',
    url='https://github.com/X-lab-3D/PANDORA/tree/master',
    project_urls={
        'Source Code': 'https://github.com/X-lab-3D/PANDORA/tree/master',
        'Issue tracker': 'https://github.com/X-lab-3D/PANDORA/issues'
    },
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    keywords='PANDORA',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    install_requires=[
       'biopython',
       'pdb2sql',
       'joblib',
       'urllib3==1.26.14',
       ],

    extras_require={
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme'],
        'test': ['pytest', 'pytest-runner', 'pytest-cov',
                 'coverage', 'coveralls', 'pycodestyle']
    },
    
    entry_points={
        'console_scripts':[
            'pandora-fetch=PANDORA.cmd_pandora:cmd_install_database',
            'pandora-create=PANDORA.cmd_pandora:cmd_create_database',
            'pandora-run=PANDORA.cmd_pandora:cmd_run_pandora',
            'pandora-wrapper=PANDORA.cmd_pandora:cmd_run_wrapper',
            ],}
)