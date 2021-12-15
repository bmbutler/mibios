"""
The NCBI Taxonomy Database
"""
from pathlib import Path
from subprocess import run
import sys

from django.conf import settings

from mibios import get_registry


REMOTE_HOST = 'ftp.ncbi.nih.gov'
REMOTE_PATH = '/pub/taxonomy/new_taxdump'
ARCHIVE_NAME = 'new_taxdump.tar.gz'
README = 'taxdump_readme.txt'

WGET = '/usr/bin/wget'
MD5SUM = '/usr/bin/md5sum'
TAR = '/bin/tar'


def get_data_source_path():
    """
    Return path to data download/source directory
    """
    try:
        return Path(settings.NCBI_TAXONOMY_SOURCE_DIR)
    except AttributeError:
        return Path.cwd()


def download_latest(dest=None):
    """
    Download latest version of the taxonomy to destination directory
    """
    if dest is None:
        dest = get_data_source_path()
    else:
        dest = Path(dest)

    if not dest.is_dir():
        dest.mkdir()
        print(f'Created directory: {dest}', file=sys.stderr)

    url = f'https://{REMOTE_HOST}:{REMOTE_PATH}/{README}'
    run([WGET, '--backups=1', url], cwd=dest, check=True)
    url = f'https://{REMOTE_HOST}:{REMOTE_PATH}/{ARCHIVE_NAME}'
    run([WGET, '--backups=1', url], cwd=dest, check=True)
    run([WGET, '--backups=1', url + '.md5'], cwd=dest, check=True)
    run([MD5SUM, '--check', ARCHIVE_NAME + '.md5'], cwd=dest, check=True)
    run([TAR, 'vxf', ARCHIVE_NAME], cwd=dest, check=True)


"""
Model names in order of FK-relational dependencies.  Models earlier in the
order must be populated before later ones so that FK fields can be set (as the
PKs must be known).  The data can be deleted in reverse order.
"""
REL_DEPEND_ORDER = [
        'gencode',
        'division',
        'deletednode',
        'taxnode',
        'taxname',
        'mergednodes',
        'host',
        'typematerialtype',
        'typematerial',
        'citation',
]


def load():
    """
    Load all data from the downloaded source

    Not loading until we know we need this:
        taxidlineage.dmp
        rankedlineage.dmp
        fullnamelineage.dmp
    """
    models = get_registry().apps['ncbi_taxonomy'].models
    for i in REL_DEPEND_ORDER:
        model = models[i]
        print(f'Loading {model._meta.model_name}...', end='', flush=True)
        rest = model.load()
        if rest:
            print(' [some unprocessed data]')
        else:
            print(' [OK]')


def erase():
    from mibios_umrad.model_utils import delete_all_objects_quickly
    for i in get_registry().apps['ncbi_taxonomy'].get_models():
        delete_all_objects_quickly(i)
