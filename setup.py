#!/usr/bin/env python3
import os.path
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
        path = os.path.sep.join([self.build_lib, NAME, '__init__.py'])
        self.announce('Patching version "{}" into: {}'
                      ''.format(get_version(), path))
        with open(path, 'a') as f:
            f.write(self.template.format(get_version()))


class BuildPyCmd(build_py):
    def run(self):
        super().run()
        self.run_command('set_version')


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
    python_requires='>=3.7',
    install_requires=[
        'Django~=2.2.0',
        'django-extensions~=2.2.0',
        'djangorestframework~=3.11.0',
        'django-tables2~=2.1.0',
        'pandas~=0.25.0',
        'biopython~=1.77',
    ],
    packages=setuptools.find_packages(),
    package_data={
        'mibios': ['templates/' + NAME + '/*.html',
                   'static/' + NAME + '/css/*.css'],
        'mibios_seq': ['templates/' + NAME + '/*.html'],
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
