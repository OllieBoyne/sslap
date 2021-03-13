from setuptools import setup
from distutils.extension import Extension
import logging
import sys

requirements = ['numpy', 'matplotlib', 'tqdm', 'scipy', 'cython']

# Set up the logging environment
logging.basicConfig()
log = logging.getLogger()

# Check Cython installation
try:
	from Cython.Build import cythonize
except ImportError:
	log.warning("Cython required to compile project. Installing now:")
	import pip
	pip.main(['install', 'cython'])
	try:
		from Cython.Build import cythonize
	except ImportError:
		log.critical(
			'Cython.Build.cythonize not found. '
			'Cython is required to build from a repo.')
		sys.exit(1)

# Extension options
include_dirs = []
try:
	import numpy
	include_dirs.append(numpy.get_include())
except ImportError:
	log.critical('Numpy and its headers are required to run setup(). Exiting')
	sys.exit(1)


opts = dict(
	include_dirs=include_dirs,
)
ext_modules = cythonize([
	Extension(
		'sslap.auction_', ['sslap/auction_.pyx'], **opts),
	Extension(
		'sslap.feasibility_', ['sslap/feasibility_.pyx'], ** opts)
]
)

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setup(
	name='sslap',
	version='0.2.3',
	description='Super Sparse Linear Assignment Problems Solver',
	long_description=long_description,
	long_description_content_type="text/markdown",
	author_email='ollieboyne@gmail.com',
	packages=['sslap'],
	url='http://github.com/OllieBoyne/sslap',
	author='Ollie Boyne',
	ext_modules=ext_modules,
	install_requires=requirements,
	license="MIT",
	keywords='super sparse linear assignment problem solve lap auction algorithm',
)

