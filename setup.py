#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding='utf8') as history_file:
    history = history_file.read()

requirements = [
    'Click',
    'humanize',
    'nibabel',
    'numpy',
    'scipy',
    'torch>=1.1',
    'torchvision',
    'tqdm',
]


def is_slicer_python():
    """
    Returns True if the code is believed to be executed from within Slicer's
    internal Python.
    """
    python_home = os.environ.get('PYTHONHOME')
    return python_home is not None and 'Slicer' in python_home


# New versions of Slicer need SimpleITK 2, but SimpleITK is preferred
# because of https://github.com/SimpleITK/SimpleITK/issues/1239
if not is_slicer_python():
    requirements.append('SimpleITK<2')


setup(
    author='Fernando Perez-Garcia',
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=(
        'Tools for loading, augmenting and writing 3D medical images'
        ' on PyTorch.'
    ),
    entry_points={
        'console_scripts': [
            'torchio-transform=torchio.cli:apply_transform',
        ],
    },
    extras_require={
        'plot': ['matplotlib'],
    },
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='torchio',
    name='torchio',
    packages=find_packages(include=['torchio', 'torchio.*']),
    setup_requires=[],
    test_suite='tests',
    tests_require=[],
    url='https://github.com/fepegar/torchio',
    version='0.18.0',
    zip_safe=False,
)
