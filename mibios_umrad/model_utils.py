"""
Model-related utilities

Separate module to avoid circular imports issues
"""
from itertools import islice
import os

from django.core.exceptions import FieldDoesNotExist
from django.db import connection, models
from django.db.transaction import atomic

from mibios.models import Model as MibiosModel, Manager as MibiosManager
from .utils import DryRunRollback, ProgressPrinter, chunker


# standard data field options
opt = dict(blank=True, null=True, default=None)  # non-char optional
ch_opt = dict(blank=True, default='')  # optional char
uniq_opt = dict(unique=True, **opt)  # unique and optional (char/non-char)
# standard foreign key options
fk_req = dict(on_delete=models.CASCADE)  # required FK
fk_opt = dict(on_delete=models.SET_NULL, **opt)  # optional FK


class LoadMixin:
    """
    Provide functionality to load data from file
    """
    @classmethod
    def get_file(cls):
        """
        Return pathlib.Path to input data file
        """
        raise NotImplementedError('must be implemented in inheriting model')

    import_file_spec = None
    """
    A list of field names or list of tuples matching file headers to field
    names.  Use list of tuples if import file has a header.  Otherwise use the
    simple list, listing the fields in the right order.
    """

    @classmethod
    def load(cls, max_rows=None, dry_run=True, sep='\t', parse_only=False):
        """
        Load data from file

        Assumes empty table
        """
        with cls.get_file().open() as f:
            print(f'File opened: {f.name}')
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            if cls.import_file_spec is None:
                raise NotImplementedError(f'import_file_spec not set in {cls}')

            if isinstance(cls.import_file_spec[0], tuple):
                # check header
                head = f.readline().rstrip('\n').split(sep)
                cols = [i[0] for i in cls.import_file_spec]
                for i, (a, b) in enumerate(zip(head, cols), start=1):
                    # ignore column number differences here, will catch later
                    if a != b:
                        raise ValueError(
                            f'Unexpected header row: first mismatch in column '
                            f'{i}: got "{a}", expected "{b}"'
                        )
                if len(head) != len(cols):
                    raise ValueError(
                        f'expecting {len(cols)} columns but got {len(head)} '
                        'in header row'
                    )

            if max_rows is None:
                file_it = f
            else:
                file_it = islice(f, max_rows)

            if parse_only:
                return cls._parse_lines(file_it, sep=sep)

            try:
                return cls._load_lines(file_it, sep=sep, dry_run=dry_run)
            except DryRunRollback:
                print('[dry run rollback]')

    @classmethod
    @atomic
    def _load_lines(cls, lines, sep='\t', dry_run=True):
        ncols = len(cls.import_file_spec)

        objs = []
        m2m_data = {}
        split = cls._split_m2m_input
        # TODO: allow fields to be skipped
        fields = [cls._meta.get_field(i[1]) for i in cls.import_file_spec]
        accession_field_name = cls.get_accession_field().name
        pp = ProgressPrinter(f'{cls._meta.model_name} records read')
        for line in pp(lines):
            obj = cls()
            m2m = {}
            row = line.rstrip('\n').split(sep)

            if len(row) != ncols:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {pp.current=} {row=} '
                    f'{ncols=}'
                )

            for i, field in enumerate(fields):
                if field.many_to_many:
                    m2m[field.name] = split(row[i])
                else:
                    if not row[i] == '':
                        # TODO: find out why leaving '' in for int fields fails
                        # ValueError @ django/db/models/fields/__init__.py:1825
                        setattr(obj, field.name, row[i])

            objs.append(obj)
            m2m_data[getattr(obj, accession_field_name)] = m2m

        if not objs:
            # empty file?
            return

        m2m_fields = list(m2m.keys())
        del m2m

        batch_size = 990  # sqlite3 max is around 999 ?
        pp = ProgressPrinter(f'{cls._meta.model_name} records written to DB')
        for batch in chunker(objs, batch_size):
            try:
                batch = cls.objects.bulk_create(batch)
            except Exception:
                print(f'ERROR saving to {cls}: batch 1st: {vars(batch[0])=}')
                raise
            pp.inc(len(batch))
            if batch[0].pk is not None:
                print(batch[0], batch[0].pk)

        del objs
        pp.finish()

        # get accession -> pk map
        accpk = dict(
            cls.objects.values_list(accession_field_name, 'pk').iterator()
        )

        # replace accession with pk in m2m data keys
        m2m_data = {accpk[i]: data for i, data in m2m_data.items()}
        del accpk

        # collecting all m2m entries
        for i in m2m_fields:
            cls._update_m2m(i, m2m_data)

        if dry_run:
            raise DryRunRollback

    @classmethod
    def _update_m2m(cls, field_name, m2m_data):
        """
        Update M2M data for one field -- helper for _load_lines()

        :param str field_name: Name of m2m field
        :param dict m2m_data:
            A dict with all fields' m2m data as produced in the load_lines
            method.
        """
        print(f'm2m {field_name}: ', end='', flush=True)
        field = cls._meta.get_field(field_name)
        model = field.related_model
        acc_field = model.get_accession_field()
        acc_field_name = acc_field.name

        # extract and flatten all accessions for field in m2m data
        accs = (i for objdat in m2m_data.values() for i in objdat[field_name])
        if acc_field.get_internal_type().endswith('IntegerField'):
            # cast to right type as needed (integers only?)
            accs = (int(i) for i in accs)
        accs = set(accs)
        print(f'{len(accs)} unique accessions in data - ', end='', flush=True)
        if not accs:
            print()
            return

        # get existing
        qs = model.objects.all()
        qs = qs.values_list(acc_field_name, 'pk')
        a2pk = dict(qs.iterator())
        print(f'known: {len(a2pk)} ', end='', flush=True)

        # save new
        new_accs = [i for i in accs if i not in a2pk]
        if new_accs:
            if model.get_fields().names == ['id', acc_field_name]:
                # model is UID/accession stub, we have all the info right here
                print(f'new: {len(new_accs)} ', end='', flush=True)
                new_related = (model(**{acc_field_name: i}) for i in new_accs)
                new_related = model.objects.bulk_create(new_related)
                print('(saved)', end='', flush=True)
            else:
                # data needs to have been added already, from other source
                print()
                raise RuntimeError(
                    f'no auto-adding new {field_name} entries: '
                    + ' '.join([f'"{i}"' for i in new_accs[:20]])
                    + '...'
                )

            # get m2m field's key -> pk mapping
            a2pk = dict(
                model.objects.values_list(acc_field_name, 'pk').iterator()
            )

        # set relationships
        rels = []  # pairs of ours and other's PKs
        for i, other in m2m_data.items():
            rels.extend(((i, a2pk[j]) for j in other[field_name]))
        Through = field.remote_field.through  # the intermediate model
        our_id_name = cls._meta.model_name + '_id'
        other_id_name = model._meta.model_name + '_id'
        through_objs = [
            Through(
                **{our_id_name: i, other_id_name: j}
            )
            for i, j in rels
        ]
        Through.objects.bulk_create(through_objs)

        print(f' ({len(through_objs)} relations saved)', end='', flush=True)
        print()

    @classmethod
    def _split_m2m_input(cls, value):
        """
        Helper to split semi-colon-separated list-field values in import file
        """
        # split and remove empties:
        items = (i for i in value.split(';') if i)
        # TODO: duplicates in input data (NAME/function column), tell Teal?
        # TODO: check with Teal if order matters or if it's okay to sort
        items = sorted(set(items))
        return items

    @classmethod
    def get_accession_field(cls):
        """
        Return the accession / human-facing UID field

        Usually this is the first non-pk unique field, going by class
        declaration order.

        Raises KeyError if no such field exists.
        """
        return [
            i for i in cls._meta.get_fields()
            if hasattr(i, 'unique') and i.unique and not i.primary_key
        ][0]

    @classmethod
    def _parse_lines(cls, lines, sep='\t'):
        ncols = len(cls.import_file_spec)

        data = []
        split = cls._split_m2m_input
        # TODO: allow fields to be skipped
        fieldnames = [i[1] for i in cls.import_file_spec]

        pp = ProgressPrinter(f'{cls._meta.model_name} records read')
        for line in pp(lines):
            row = line.rstrip('\n').split(sep)

            if len(row) != ncols:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {pp.current=} {row=} '
                    f'{ncols=}'
                )

            rec = {}
            for fname, value in zip(fieldnames, row):
                try:
                    field = cls._meta.get_field(fname)
                except FieldDoesNotExist:
                    # caller must handle values from m2m field (if any)
                    pass
                else:
                    if field.many_to_many:
                        value = split(value)

                rec[fname] = value

            data.append(rec)
        return data


class Manager(MibiosManager):
    def bulk_create(self, objs, batch_size=None, ignore_conflicts=False,
                    progress_text=None, progress=True):
        # Value-added bulk_create with proggress metering
        if batch_size is None:
            batch_size = 990  # sqlite3 bulk max size is just below 1000?

        if progress:
            if progress_text is None:
                progress_text = \
                    f'{self.model._meta.verbose_name_plural} records created'
            pp = ProgressPrinter(progress_text)

        for batch in chunker(objs, batch_size):
            try:
                batch = super().bulk_create(
                    batch,
                    ignore_conflicts=ignore_conflicts,
                )
            except Exception:
                print(f'ERROR saving to {self.model}: batch 1st: '
                      f'{vars(batch[0])=}')
                raise
            if progress:
                pp.inc(len(batch))

        if progress:
            pp.finish()


class Model(MibiosModel, LoadMixin):
    history = None
    objects = Manager()

    class Meta:
        abstract = True


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
