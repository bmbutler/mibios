"""
Model-related utilities

Separate module to avoid circular imports issues
"""
from collections import UserDict
from itertools import islice
import os

from django.core.exceptions import FieldDoesNotExist
from django.db import connections, models, router
from django.db.transaction import atomic, set_rollback

from mibios.models import Model as MibiosModel
from .manager import Manager
from .utils import ProgressPrinter


# standard data field options
opt = dict(blank=True, null=True, default=None)  # non-char optional
ch_opt = dict(blank=True, default='')  # optional char
uniq_opt = dict(unique=True, **opt)  # unique and optional (char/non-char)
# standard foreign key options
fk_req = dict(on_delete=models.CASCADE)  # required FK
fk_opt = dict(on_delete=models.SET_NULL, **opt)  # optional FK


class digits(UserDict):
    """
    Abbreviation for max_digits and decimal_places DecimalField keyword args

    Usage:  foo = models.DecimalField(**digits(8, 3))
    """
    def __init__(self, max_digits, decimal_places):
        super().__init__(max_digits=max_digits, decimal_places=decimal_places)


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
            m2m_data[obj.get_accession_single()] = m2m

        if not objs:
            # empty file?
            return

        m2m_fields = list(m2m.keys())
        del m2m

        cls.objects.bulk_create(objs)
        del objs

        # get accession -> pk map
        accpk = dict(
            cls.objects
            .values_list(cls.get_accession_field_single(), 'pk')
            .iterator()
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
        acc_field = model.get_accession_field_single()

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
        qs = qs.values_list(acc_field.name, 'pk')
        a2pk = dict(qs.iterator())
        print(f'known: {len(a2pk)} ', end='', flush=True)

        new_accs = [i for i in accs if i not in a2pk]
        if new_accs:
            # save additional remote side objects
            # NOTE: this will only work for those models for which the supplied
            # information (accession, source model and field) is sufficient,
            # might require overriding create_from_m2m_input().
            print(f'new: {len(new_accs)}', end='')
            if a2pk:
                # if old ones exist, then display some of the new ones, this
                # is for debugging, in case these ought to be known
                print(' ', ', '.join([str(i) for i in islice(new_accs, 5)]),
                      '...')
            else:
                print()

            model.objects.create_from_m2m_input(
                new_accs,
                source_model=cls,
                src_field_name=field_name,
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
        Manager.bulk_create_wrapper(Through.objects.bulk_create)(through_objs)

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


class Model(MibiosModel):
    history = None
    objects = Manager()

    class Meta:
        abstract = True
        default_manager_name = 'objects'

    @classmethod
    def get_accession_fields(cls):
        """
        Return the accession / human-facing UID fields

        Returns a tuple of fields.

        Usually this is the first non-pk unique field, going by class
        declaration order.

        If no such field exist, then return the first tuple of fields declared
        as "unique_together" if such a declaration took place.

        Raises KeyError in all other cases.
        """
        unique_fields = [
            i for i in cls._meta.get_fields()
            if hasattr(i, 'unique') and i.unique and not i.primary_key
        ]
        if unique_fields:
            return (unique_fields[0], )

        if cls._meta.unique_together:
            return tuple((
                cls._meta.get_field(i)
                for i in cls._meta.unique_together[0]
            ))

        raise LookupError(f'model {cls} has no unique/unique_together fields')

    @classmethod
    def get_accession_field_single(cls):
        """
        Return the accession / natural key / human-facing UID field

        Use this instead of get_accession_fields() if the calling site can only
        handle a single such accession field.  If this method is called on a
        model that has multiple "unique_together" fields a RuntimeError is
        raised.
        """
        fields = cls.get_accession_fields()
        if len(fields) > 1:
            raise RuntimeError('model has multiple "unique_together" fields')
        return fields[0]

    @classmethod
    def get_accession_lookups(cls):
        """
        Get lookups/accessors to retrieve accession values

        Similar to get_accesion_fields, but returns the field names and in case
        of foreign key fields, returns the lookup to the FK's models accession.
        """
        ret = []
        for i in cls.get_accession_fields():
            attr_name = i.name
            if i.many_to_one:
                attr_name += '_id'
            ret.append(attr_name)
        return tuple(ret)

    @classmethod
    def get_accession_lookup_single(cls):
        """
        Get lookup/accessor/fieldname to retrieve the accession value

        Similar to get_accession_field_single, but returns the field name or in
        case of a foreign key field, returns the lookup to the FK's models
        accession.

        Use this method instead of get_accession_lookups() if the calling site
        can only handle a single such lookup.

        Will raise a RuntimeError of called on a model with multiple
        "unique_together" accession fields.
        """
        names = cls.get_accession_lookups()
        if len(names) > 1:
            raise RuntimeError(
                f'model {cls} has multiple "unique_together" fields'
            )
        return names[0]

    def get_accessions(self):
        """
        Return instance-identifying value(s)

        For FK fields this returns the pk
        """
        accs = []
        for field in self.get_accession_fields():
            if field.many_to_one:
                attr_name = field.name + '_id'
            else:
                attr_name = field.name

            accs.append(getattr(self, attr_name))
        return tuple(accs)

    def get_accession_single(self):
        """
        Return accession/natural key value

        For FK fields this returns the pk.  Use this method instead of
        get_accessions() is the calling site can not handle tuples but only a
        single value.

        Raises a RuntimeError if called on a model with multiple
        "unique_together" accession fields.
        """
        values = self.get_accessions()
        if len(values) > 1:
            raise RuntimeError('model has multiple "unique_together" fields')
        return values[0]

    def set_accession(self, *values):
        """
        Set instance-identifying value(s)
        """
        for field, value in zip(self.get_accession_fields(), values):
            if field.many_to_one:
                attr_name = field.name + '_id'
            else:
                attr_name = field.name

            setattr(self, attr_name, value)

    @classmethod
    def get_search_field(cls):
        """
        Return the default search field

        Usually this is the first (non-pk) unique id fields.  Models that don't
        have such a field must overwrite this method.
        """
        return cls.get_accession_field_single()

    def get_external_url(self):
        """
        get external URL for this record

        Inheriting models should overwrite this to suit their needs.  The
        default implementation returns None.
        """
        pass


class VocabularyModel(Model):
    """
    A list of controlled vocabulary
    """
    max_length = 64
    entry = models.CharField(max_length=max_length, unique=True, blank=False)

    class Meta(Model.Meta):
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
    db_alias = router.db_for_write(model)
    with connections[db_alias].cursor() as cur:
        cur.execute(f'delete from {model._meta.db_table}')
        res = cur.fetchall()
    if res != []:
        raise RuntimeError('expected empty list returned')
