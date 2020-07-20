#!/usr/bin/env python3
import setuptools


NAME = 'mibios'


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version='0.0.2',
    author_email='heinro@med.umich.edu',
    description='Microbiome data project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://sites.google.com/a/umich.edu/the-schmidt-lab/home',
    python_requires='>=3.8',
    install_requires=[
        'django',
        'django-tables2',
        'django-extensions',
        'pandas',
    ],
    packages=setuptools.find_packages(),
    package_data={
        'mibios': ['templates/' + NAME + '/*.html',
                   'static/' + NAME + '/css/*.css'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Framework :: Django',
        # 'License :: None',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
