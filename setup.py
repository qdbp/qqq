import sys
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# semver with automatic minor bumps keyed to unix time
__version__ = "0.6.1512668342"


# hack from
# https://stackoverflow.com/questions/2379898/make-distutils-look-for-numpy-header-files-in-the-correct-place
# to make numpy header finder usable during setup
class build_ext_hack(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        from numpy.distutils.misc_util import get_numpy_include_dirs

        self.include_dirs.extend(get_numpy_include_dirs())
        super().run()


CX_MODS: List[Extension] = []

if sys.platform == "linux":
    CX_MODS += [
        Extension(
            "vnv.screenshot",
            ["vnv/cx/screenshot.c"],
            extra_link_args=["-lX11"],
            extra_compile_args=["-O3", "-mtune=native", "-march=native"],
        )
    ]


setup(
    # info
    name="vnv",
    version=__version__,
    # contents
    packages=["vnv"],
    ext_modules=CX_MODS,
    # requirements
    setup_requires=["numpy"],
    install_requires=["numpy"],
    extras_require={
        "test": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-xdist",
            "pytest-testmon",
            "filelock",
            "scikit-learn",
        ],
        "keras": ["keras", "tensorflow", "matplotlib"],
        "web": ["requests", "regex", "lxml", "tldextract"],
        "docs": [
            "sphinx",
            "sphinxcontrib-napoleon",
            "sphinxcontrib-trio",
            "docformatter",
        ],
    },
    # details
    cmdclass={"build_ext": build_ext_hack},
)
