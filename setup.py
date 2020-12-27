
# OPTIONAL. HERE, CHANGE USE_CYTHON TO TRUE IF YOU HAVE EDITED AUCTION.PYX, AND WANT THAT TO BE COMPILED INSTEAD
USE_CYTHON = False

if USE_CYTHON:
	try:
		from Cython.Distutils import build_ext
	except ImportError:
		raise ImportError("USE_CYTHON set to True, but Cython installation not found.")

import sys
from setuptools import setup
from setuptools import Extension


if sys.version_info[0] == 2:
	raise Exception('Python 2 is not supported')

cmdclass = { }
ext_modules = [ ]

# compile cython
if USE_CYTHON:
	ext_modules += [Extension('sslap.auction', ['sslap/auction.pyx'])]
	cmdclass.update({'build_ext': build_ext})
else:
	ext_modules += [Extension('sslap.auction', ['sslap/auction.c'])]

setup(
    name='sslap',
    version='0.1',
    description='Super Sparse Linear Assignment Problems Solver',
    author='Ollie Boyne',
    author_email='ollieboyne@gmail.com',
    url='http://github.com/OllieBoyne/sslap',
    packages=[ 'sslap', ],
    package_dir={
        'sslap' : 'python3/simplerandom',
    },
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    license="MIT",
    keywords='super sparse linear assignment problem solve lap auction algorithm',
)