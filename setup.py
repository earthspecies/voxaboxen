#!/usr/bin/env python3

from pathlib import Path

from setuptools import setup

NAME = 'voxaboxen'
DESCRIPTION = (
    'Bioacoustics event detection deep learning framework.'
    ' Supports training and evaluation using Raven annotations and simple config files.'
    ' For training, please directly clone the github repository.')

URL = 'https://github.com/earthspecies/voxaboxen'
EMAIL = 'benjamin@earthspecies.org'
AUTHOR = 'Benjamin Hoffman, Maddie Cusimano'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = "0.1.0"

HERE = Path(__file__).parent

REQUIRED = [
    'pandas>=2.0.2',
    'librosa>=0.10.0',
    'soundfile>=0.12.1',
    'tqdm>=4.65.0',
    'numpy>=1.24.3',
    'plumbum>=1.8.2',
    'PyYAML>=6.0',
    'intervaltree>=3.1.0',
    'torch>=2.0.1',
    'torchaudio>=2.0.1',
    'einops>=0.6.1',
    'scipy>=1.10.1',
    'matplotlib>=3.7.1',
    'seaborn>=0.12.2',
    'mir_eval>=0.7',
]

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['voxaboxen'],
    install_requires=REQUIRED,
    include_package_data=True,
    license='GNU Affero General Public License v3.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)
