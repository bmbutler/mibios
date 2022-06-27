"""
Stuff related to search and search suggestions
"""
from functools import cache
from logging import getLogger

from django.conf import settings
from django.db import connections, router
from django.db.backends.signals import connection_created
from django.db.utils import OperationalError
from django.dispatch import receiver

from mibios_omics import get_sample_model
from mibios_umrad.models import (
    CompoundRecord, CompoundName, FunctionName, Location, Metal,
    FuncRefDBEntry, ReactionRecord, Taxon, Uniprot, UniRef100,
)

log = getLogger(__name__)


spellfix_models = [
    CompoundRecord, ReactionRecord, CompoundName,
    FunctionName, Location, Metal, FuncRefDBEntry, Taxon,
]
searchable_models = spellfix_models + [
    UniRef100, Uniprot, get_sample_model(),
]
SPELLFIX_TABLE = 'searchterms'


@cache
def get_connection(for_write=False):
    """
    Gets the DB connection for accessing the spellfix tables

    This basically assumes that all spellfixed models must be on the same DB
    """
    if for_write:
        alias = router.db_for_write(spellfix_models[0])
    else:
        alias = router.db_for_read(spellfix_models[0])
    return connections[alias]


@receiver(connection_created)
def load_sqlite_spellfix_extension(
    connection=None,
    spellfix_ext_path=None,
    **kwargs,
):
    """
    Load spellfix extension for sqlite3 DBs
    """
    if connection is None:
        raise ValueError('connection parameter must not be None')

    if spellfix_ext_path:
        path = spellfix_ext_path
    else:
        path = settings.SPELLFIX_EXT_PATH

    if connection.vendor == 'sqlite' and path:
        connection.connection.load_extension(path)
        log.info('sqlite3 spellfix extension loaded')


def update_spellfix(spellfix_ext_path=None):
    """
    Populate search spellfix suggestions table

    :param str spellfix_ext_path:
        Provide path to spellfix shared object in case.  With this the spellfix
        tables can be populated before search suggestion get enabled via
        settings.

    This needs to run once, before get_suggestions() can be called.
    """
    connection = get_connection(for_write=True)

    if spellfix_ext_path is not None:
        connection.ensure_connection()
        load_sqlite_spellfix_extension(
            connection=connection,
            spellfix_ext_path=spellfix_ext_path,
        )

    with connection.cursor() as cur:
        cur.execute('BEGIN')
        cur.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS '
                    f'{SPELLFIX_TABLE} USING spellfix1')
        cur.execute(f'DELETE FROM {SPELLFIX_TABLE}_vocab')
        log.info('spellfix table deleted')
        for model in spellfix_models:
            cur.execute(
                'INSERT INTO {spellfix_table}(word) SELECT {field} '
                'FROM {table} WHERE {field} NOT IN '
                '(SELECT word FROM {spellfix_table}_vocab)'.format(
                    spellfix_table=SPELLFIX_TABLE,
                    field=model.get_search_field().name,
                    table=model._meta.db_table,
                )
            )
            log.info(f'added to search suggestions: {model._meta.model_name}')
        cur.execute('COMMIT')


def get_suggestions(query):
    """ get top 20 spellfix suggestions """
    if not settings.SPELLFIX_EXT_PATH:
        return []

    with get_connection().cursor() as cur:
        try:
            cur.execute(
                f'select word from {SPELLFIX_TABLE} where word match %s '
                f'and scope=4',  # small speed-up
                [query]
            )
        except OperationalError as e:
            if e.args[0] == f'no such table: {SPELLFIX_TABLE}':
                log.warning('Search suggestions were not set up, '
                            'update_spellfix() needs to be run!')
            else:
                raise
        return [i[0] for i in cur.fetchall()]
