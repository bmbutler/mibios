#!/usr/bin/env python3
from pathlib import Path
import subprocess

import setuptools
from setuptools.command.build_py import build_py


NAME = 'mibios'

_version = None


def get_version():
    global _version
    if _version is None:
        _version = subprocess.run(
            ['git', 'describe', '--tags', '--always', '--match', 'v*'],
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.decode().strip().lstrip('v')
        # make PEP 440 conform (as local version) and normalize
        _version = _version.replace('-', '+', 1).replace('-', '.')
    return _version


class SetVersionCmd(setuptools.Command):
    template = '__version__ = \'{}\'  # added by `setyp.py build`\n'

    def initialize_options(self):
        self.build_lib = None

    def finalize_options(self):
        self.set_undefined_options('build', ('build_lib', 'build_lib'))

    def run(self):
        path = Path(self.build_lib) / NAME / '__init__.py'
        self.announce('Patching version "{}" into: {}'
                      ''.format(get_version(), path))
        with path.open('a') as f:
            f.write(self.template.format(get_version()))


class BuildPyCmd(build_py):
    def run(self):
        super().run()
        self.run_command('set_version')


def get_package_data():
    """
    Return a dict top-level-package -> list of globs of data files
    """
    # get top-level packages:
    plist = [i for i in setuptools.find_packages() if '.' not in i]
    data = {}
    paths = [  # django app data files / normal directory layout
        'templates/*.html',
        'templates/{app}/*.html',
        'static/{app}/css/*.css',
        'static/{app}/js/*.js',
        'static/{app}/img/*.png',
    ]
    for app in plist:
        # assume app is django app
        for p in paths:
            p = p.format(app=app)
            if list(Path(app).glob(p)):
                if app not in data:
                    data[app] = []
                data[app].append(p)
    return data


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=get_version(),
    author_email='heinro@med.umich.edu',
    description='Microbiome data project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://sites.google.com/a/umich.edu/the-schmidt-lab/home',
    python_requires='>=3.10',
    install_requires=[
        'biopython~=1.80',
        'defusedxml~=0.7',
        'Django~=3.2.0',
        'django-extensions~=3.2.0',
        'djangorestframework~=3.14.0',
        'django-tables2~=2.4.0',
        'matplotlib~=3.6.0',
        'pandas~=1.3.0',
        'psycopg2~=2.9.0',
        'xlrd~=1.2',
        'zipstream~=1.1.0',
    ],
    packages=setuptools.find_packages(),
    package_data=get_package_data(),
    entry_points={
        'console_scripts': [
            'manage_' + NAME + '=' + NAME + '.ops:manage',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Framework :: Django',
        # 'License :: None',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    cmdclass={
        'set_version': SetVersionCmd,
        'build_py': BuildPyCmd,
    },
)
