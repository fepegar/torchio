#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'nibabel',
    'numpy',
    'SimpleITK',
    'torch',
    'torchvision',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Fernando Perez-Garcia",
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Operating System :: OS Independent',
    ],
    description="Dataset tools for medical images",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='torchio',
    name='torchio',
    packages=find_packages(include=['torchio']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fepegar/torchio',
    version='0.1.0',
    zip_safe=False,
)
