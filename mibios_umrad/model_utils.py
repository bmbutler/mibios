"""
Model-related utilities

Separate module to avoid circular imports issues
"""
from itertools import islice
import os

from django.core.exceptions import FieldDoesNotExist
from django.db import connection, models
from django.db.transaction import atomic, set_rollback

from mibios.models import Model as MibiosModel, Manager as MibiosManager
from .utils import ProgressPrinter


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
    def load(cls, max_rows=None, start=0, dry_run=False, sep='\t',
             parse_only=False):
        """
        Load data from file

        :param int start:
            0-based input file line (not counting any header) from which to
            start loading, equivalent to number of non-header lines skipped.
        :param bool parse_only:
            Return each row as a dict and don't use the general loader logic.

        May assume empty table ?!?
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

            pp = ProgressPrinter(f'{cls._meta.model_name} records read')
            f = pp(f)

            if max_rows is None and start == 0:
                file_it = f
            else:
                file_it = islice(f, start, start + max_rows)

            if parse_only:
                return cls._parse_lines(file_it, sep=sep)

            return cls._load_lines(file_it, sep=sep, dry_run=dry_run)

    @classmethod
    @atomic
    def _load_lines(cls, lines, sep='\t', dry_run=False):
        ncols = len(cls.import_file_spec)

        objs = []
        m2m_data = {}
        split = cls._split_m2m_input
        # TODO: allow fields to be skipped
        fields = [cls._meta.get_field(i) for _, i in cls.import_file_spec]
        accession_field_name = cls.get_accession_field().name
        for line in lines:
            obj = cls()
            m2m = {}
            row = line.rstrip('\n').split(sep)

            if len(row) != ncols:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {row=} '
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

        cls.objects.bulk_create(objs)
        del objs

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
            set_rollback(True)

    @classmethod
    def _update_m2m(cls, field_name, m2m_data):
        """
        Update M2M data for one field -- helper for _load_lines()

        :param str field_name: Name of m2m field
        :param dict m2m_data:
            A dict with all fields' m2m data as produced in the load_lines
            method.
        :param callable create_fun:
            Callback function that creates a missing instance on the other side
            of the relation.
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

        new_accs = [i for i in accs if i not in a2pk]
        if new_accs:
            # save additional remote side objects
            # NOTE: this will only work for those models for which the supplied
            # information (accession, source model and field) is sufficient,
            # might require overriding create_from_m2m_input().
            print(f'new: {len(new_accs)} ')
            model.objects.create_from_m2m_input(
                new_accs,
                source_model=cls,
                source_field_name=field_name,
            )

            # get m2m field's key -> pk mapping
            a2pk = dict(qs.iterator())
        else:
            print()

        # set relationships
        rels = []  # pairs of ours and other's PKs
        for i, other in m2m_data.items():
            rels.extend((
                (i, a2pk[acc_field.to_python(j)])
                for j in other[field_name]
            ))
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
        col_keys = [i for _, i in cls.import_file_spec]

        for line in lines:
            row = line.rstrip('\n').split(sep)

            if len(row) != ncols:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {row=} '
                    f'{ncols=}'
                )

            rec = {}
            for key, value in zip(col_keys, row):
                if key is None:
                    continue
                try:
                    field = cls._meta.get_field(key)
                except FieldDoesNotExist:
                    # caller must handle values from m2m field (if any)
                    pass
                else:
                    if field.many_to_many:
                        value = split(value)

                rec[key] = value

            data.append(rec)
        return data


class Manager(MibiosManager):
    def bulk_create(self, objs, batch_size=None, ignore_conflicts=False,
                    progress_text=None, progress=True):
        """
        Value-added bulk_create with batching and progress metering

        Does not return the object list like super().bulk_create() does.
        """
        if not progress:
            super().bulk_create(
                objs,
                ignore_conflicts=ignore_conflicts,
                batch_size=batch_size,
            )
            return

        if batch_size is None:
            # sqlite has a variable-per-query limit of 999; it's not clear how
            # one runs against that; it seems that multiple smaller INSERTs are
            # being used automatically.  So, until further notice, only have a
            # default batch size for progress metering here.
            batch_size = 999

        if progress_text is None:
            progress_text = \
                f'{self.model._meta.verbose_name} records created'

        pp = ProgressPrinter(progress_text)
        objs = iter(objs)

        while True:
            batch = list(islice(objs, batch_size))
            if not batch:
                break
            try:
                super().bulk_create(
                    batch,
                    ignore_conflicts=ignore_conflicts,
                )
            except Exception:
                print(f'ERROR saving to {self.model}: batch 1st: '
                      f'{vars(batch[0])=}')
                raise
            pp.inc(len(batch))

        pp.finish()

    def create_from_m2m_input(self, values, source_model, source_field_name):
        """
        Store objects from accession or similar value (and context, as needed)

        :param list values:
            List of accessions
        :param Model source_model:
            The calling model, the model on the far side of the m2m relation.
        :param str source_field_name:
            Name of the m2m field pointing to us

        Inheriting classes should overwrite this method if more that the
        accession or other unique ID field is needed to create the instances.
        The basic implementation will assign the given value to the accession
        field, if one exists, or to the first declared unique field, other
        that the standard id AutoField.

        This method is responsible to set all required fields of the model,
        hence the default version should only be used with controlled
        vocabulary or similarly simple models.
        """
        accession_field_name = self.model.get_accession_field().name
        model = self.model
        objs = (model(**{accession_field_name: i}) for i in values)
        return self.bulk_create(objs)


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
