"""
Model-related utilities

Separate module to avoid circular imports issues
"""
from django.db import connection, models

from mibios.models import Model


# standard data field options
opt = dict(blank=True, null=True, default=None)  # non-char optional
ch_opt = dict(blank=True, default='')  # optional char
uniq_opt = dict(unique=True, **opt)  # unique and optional (char/non-char)
# standard foreign key options
fk_req = dict(on_delete=models.CASCADE)  # required FK
fk_opt = dict(on_delete=models.SET_NULL, **opt)  # optional FK


class VocabularyModel(Model):
    """
    A list of controlled vocabulary
    """
    max_length = 64
    entry = models.CharField(max_length=max_length, unique=True, blank=False)

    class Meta:
        abstract = True
        ordering = ['entry']

    def __str__(self):
        return self.entry


def delete_all_objects_quickly(model):
    """
    Efficiently delete all objects of a model

    This is a debugging/testing aid.  The usual Model.objects.all().delete() is
    way slower for large tables.
    """
    with connection.cursor() as cur:
        cur.execute(f'delete from {model._meta.db_table}')
        res = cur.fetchall()
    if res != []:
        raise RuntimeError('expected empty list returned')
