from setuptools import setup, Extension
import numpy.distutils.misc_util as ndist_misc


# semver with automatic minor bumps keyed to unix time
__version__ = '0.6.1512579791'


cx_mods = [
    Extension(
        "screenshot",
        ["vnv/cx/screenshot.c"],
        extra_link_args=['-lX11'],
        extra_compile_args=[
            '-O3', '-mtune=native', '-march=native'
        ],
        include_dirs=ndist_misc.get_numpy_include_dirs(),
    )
]

setup(
    name='vnv',
    version=__version__,
    packages=['vnv'],
    ext_modules=cx_mods,
)
