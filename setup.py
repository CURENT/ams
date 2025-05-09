import sys
import os
from collections import defaultdict

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


def parse_requires(filename):
    with open(os.path.join(here, filename)) as requirements_file:
        reqs = [
            line for line in requirements_file.read().splitlines()
            if not line.startswith('#')
        ]
    return reqs


def get_extra_requires(filename, add_all=True):
    """
    Build ``extras_require`` from an invert requirements file.

    See:
    https://hanxiao.io/2019/11/07/A-Better-Practice-for-Managing-extras-require-Dependencies-in-Python/
    """

    with open(os.path.join(here, filename)) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if '#' in k:
                    if k.count("#") > 1:
                        raise ValueError("Invalid line: {}".format(k))

                    k, v = k.split('#')
                    tags.update(vv.strip() for vv in v.split(','))

                # Do not add the package name itself as a tag
                for t in tags:
                    extra_deps[t].add(k.strip())

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


extras_require = get_extra_requires("requirements-dev.txt")

# --- update `extras_conda` to include packages only available in PyPI ---

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
    packages=find_packages(exclude=[]),
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
    install_requires=parse_requires('requirements.txt'),
    extras_require=extras_require,
    license='GNU General Public License v3 or later (GPLv3+)',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Environment :: Console',
    ],
)
