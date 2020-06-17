#!/usr/bin/env python3
import os

import django
from django.core.management import call_command

import setuptools
from setuptools.command.build_py import build_py


NAME = 'mibios'


class CollectStaticCmd(setuptools.Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        call_command("collectstatic", interactive=False)


class BuildPyCmd(build_py):
    def run(self):
        self.run_command('collectstatic')
        super().run()


os.environ['DJANGO_SETTINGS_MODULE'] = NAME + '.ops.settings'
django.setup()

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version='0.0.1',
    author_email='heinro@med.umich.edu',
    description='Microbiome data project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://sites.google.com/a/umich.edu/the-schmidt-lab/home',
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    package_data={
        'mibios': ['templates/' + NAME + '/*.html',
                'static/' + NAME + '/css/*.css'],
    },
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
        'collectstatic': CollectStaticCmd,
        'build_py': BuildPyCmd,
    },
)
