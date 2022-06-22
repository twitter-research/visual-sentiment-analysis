"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
import os
import io
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()

readme = read('README.md')
VERSION = '0.0.0'

requirements = [
    'torch==1.2.0',
    'torchvision==0.4.0',
    'numpy==1.22.0',
    'Pillow>=7.1.0',
    'pandas==0.24.0',
    'requests==2.22.0',
]

setup(
    # Metadata
    name='visual-sentiment-analysis',
    version=VERSION,
    author='Jose Caballero',
    author_email='jcaballero@twitter.com',
    url='https://github.com/twitter-research/visual-sentiment-analysis',
    description='A library for visual sentiment analysis via emoji prediction.',
    long_description=readme,
    license='Apache 2',

    # Package info
    packages=find_packages(exclude=('tests',)),

    zip_safe=True,
    install_requires=requirements,
)
