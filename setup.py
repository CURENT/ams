from setuptools import setup, find_packages

import versioneer

setup(
    name='ltbams',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where="."),
)
