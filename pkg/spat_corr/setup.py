from distutils.core import setup, Extension
import numpy

mod = Extension('NetworkMeasures',
    include_dirs = [numpy.get_include()],
    sources = ['NetworkMeasures.c'],
#    extra_compile_args=['-fopenmp'],
#    extra_link_args=['-lgomp']
)

setup (name = 'NetworkMeasures',
    version = '0.3',
    description = 'This module provides network measures as functions.',
    author = 'Aljoscha Rheinwalt',
    author_email = 'aljoscha@pik-potsdam.de',
    ext_modules = [mod]
)
