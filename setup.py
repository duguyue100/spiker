"""Setup script for the simretina package.

Author: Yuhuang Hu
Email : yuhuang.hu@uzh.ch
"""

from setuptools import setup
from setuptools import find_packages

classifiers = """
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Image Recognition
Topic :: Scientific/Engineering :: Neuromorphic Engineering
Topic :: Scientific/Engineering :: Computer Vision
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

try:
    from spiker import __about__
    about = __about__.__dict__
except ImportError:
    about = dict()
    exec(open("spiker/__about__.py").read(), about)

setup(
    name='spiker',
    version=about["__version__"],

    author=about["__author__"],
    author_email=about["__author_email__"],

    url=about["__url__"],

    packages=find_packages(),

    classifiers=list(filter(None, classifiers.split('\n'))),
    description="Spiker - Python framework for event-driven processing."
)
