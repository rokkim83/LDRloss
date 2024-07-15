from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='geth2d_cython',
    ext_modules=cythonize("geth2d.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)