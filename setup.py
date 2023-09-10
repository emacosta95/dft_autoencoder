#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='dft_autoencoder',
    version='0.0.0',
    description='TODO',
    author='Emanuele Costa',
    author_email='emanuele.costa@unicam.it',
    url='https://github.com/emacosta95/dft_autoencoder',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

