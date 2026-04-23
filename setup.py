import sys
import os

from setuptools import find_packages, setup

import versioneer

if sys.version_info < (3, 6):
    error = """
ams does not support Python <= {0}.{1}.
Python 3.6 and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(3, 6)
    sys.exit(error)

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


setup(
    name='ltbams',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python software for scheduling modeling and co-simulation with dynanics.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Jinning Wang",
    author_email='jinninggm@gmail.com',
    url='https://github.com/CURENT/ams',
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={
        'console_scripts': [
            'ams = ams.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'ltbams': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    license='GNU General Public License v3 or later (GPLv3+)',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Environment :: Console',
    ],
)
