from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = "tlc",
      ext_modules = cythonize('tlc/colors.pyx'),  # accepts a glob pattern
    )
