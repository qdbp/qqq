from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("qqc", ["qqc.c"],
                           extra_link_args=['-lX11'],
                           extra_compile_args=['-O2', '-mtune=native',
                           '-march=native'])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
