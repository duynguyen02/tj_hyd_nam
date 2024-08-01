from setuptools import find_packages, setup
from codecs import open
from os import path

from tj_hyd_nam import __version__, __author__

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tj_hyd_nam',
    packages=find_packages(include=['tj_hyd_nam']),
    version=__version__,
    description='Python implementation of NedborAfstromnings Model (NAM) lumped rainfall–runoff model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    install_requires=[
        'contourpy==1.2.1',
        'cycler==0.12.1',
        'fonttools==4.53.1',
        'kiwisolver==1.4.5',
        'matplotlib==3.9.1',
        'numpy==2.0.1',
        'packaging==24.1',
        'pandas==2.2.2',
        'pillow==10.4.0',
        'pyparsing==3.1.2',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'scipy==1.14.0',
        'six==1.16.0',
        'tzdata==2024.1',
    ]
)
