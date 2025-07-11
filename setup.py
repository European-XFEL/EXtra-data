#!/usr/bin/env python
import os.path as osp
import re
from setuptools import setup, find_packages
import sys


def get_script_path():
    return osp.dirname(osp.realpath(sys.argv[0]))


def read(*parts):
    return open(osp.join(get_script_path(), *parts)).read()


def find_version(*parts):
    vers_file = read(*parts)
    match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"', vers_file, re.M)
    if match is not None:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name="EXtra-data",
      version=find_version("extra_data", "__init__.py"),
      author="European XFEL GmbH",
      author_email="da-support@xfel.eu",
      maintainer="Thomas Michelat",
      project_urls={
          'Documentation': 'https://extra-data.readthedocs.io/en/latest/',
          'Release notes': 'https://extra-data.readthedocs.io/en/latest/changelog.html',
          'Issues': 'https://github.com/European-XFEL/EXtra-data/issues',
          'Source': 'https://github.com/European-XFEL/EXtra-data',
      },
      description="Tools to read and analyse data from European XFEL ",
      long_description=read("README.md"),
      long_description_content_type='text/markdown',
      license="BSD-3-Clause",
      packages=find_packages(),
      package_data={
          'extra_data.tests': ['dssc_geo_june19.h5', 'lpd_mar_18.h5'],
      },
      entry_points={
          "console_scripts": [
              "lsxfel = extra_data.cli.lsxfel:main",
              "karabo-bridge-serve-files = extra_data.cli.serve_files:main",
              "karabo-bridge-serve-run = extra_data.cli.serve_run:main",
              "extra-data-validate = extra_data.validation:main",
              "extra-data-make-virtual-cxi = extra_data.cli.make_virtual_cxi:main",
              "extra-data-locality = extra_data.locality:main",
          ],
      },
      install_requires=[
          'h5py>=2.10',
          'matplotlib',
          'numpy',
          'packaging',
          'pandas',
          'xarray',
          'pyyaml',
      ],
      extras_require={
          'bridge': [
              'karabo-bridge >=0.6',
              'psutil',
          ],
          'complete': [
              'dask[array]',
              'extra_data[bridge]',
              'tomli; python_version < "3.11"',
              'zlib_into >=0.4',
          ],
          'docs': [
              'extra_data[bridge]',  # For autodoc of ZMQStreamer
              'ipython',  # For nbsphinx syntax highlighting
              'nbsphinx',
              'sphinx',
              'sphinxcontrib_github_alt',
          ],
          'test': [
              'cloudpickle',
              'coverage',
              'extra_data[complete]',
              'nbval',
              'pytest',
              'pytest-cov',
              'testpath',
          ]
      },
      python_requires='>=3.10',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Physics',
      ]
)
