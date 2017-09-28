from setuptools import setup


# semver with automatic minor bumps keyed to unix time
__version__ = '0.3.1506618670'


if __name__ == '__main__':

    setup(
        name='vnv',
        version=__version__,
        pacakges=['vnv'],
    )
