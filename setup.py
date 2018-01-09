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

__version__ = "0.1.0-alpha.4"
__author__ = "Yuhuang Hu"
__author_email__ = "yuhuang.hu@ini.uzh.ch"
__url__ = "https://github.com/duguyue100/spiker"

setup(
    name='spiker',
    version=__version__,

    author=__author__,
    author_email=__author_email__,

    url=__url__,

    install_requires=["numpy",
                      "scipy",
                      "future",
                      "multiprocess"],
    extras_require={
          "h5py": ["h5py"],
          "deep-learning": ["Keras"],
          "visualize": ["pydot>=1.2.0"],
          "robotics": ["rosbag"]
      },

    packages=find_packages(),

    classifiers=list(filter(None, classifiers.split('\n'))),
    description="Spiker - Python framework for event-driven processing."
)
