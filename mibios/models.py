from collections import namedtuple, OrderedDict
import re
import sys

from django.apps import apps
from django.contrib.auth.models import User
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core import serializers
from django.urls import reverse
from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import models, transaction

import pandas

from .utils import getLogger


log = getLogger(__name__)


class NaturalKeyLookupError(Exception):

    """
    Raised when a natural lookup fails to resolve

    Handle like a user input error, but beware they might be bugs in the
    natural lookup handling code
    """
    pass


class Q(models.Q):
    """
    A thin wrapper around Q to handle natural lookups

    Ideally, "natural" should be a custom Lookup and then handled closer to
    the lookup-to-SQL machinery but it's not clear how to build a lookup that
    can be model-dependent expressed purely in terms of other lookups and
    doesn't touch SQL in at all.  Hence, we re-write natural lookups before
    they get packaged into the Q object.
    """
    def __init__(self, *args, model=None, **kwargs):
        # handling natural lookups is only done if the model is provided,
        # since we need to know to which model the Q is relative to
        if model is not None:
            kwargs = model.handle_natural_lookups(**kwargs)
        super().__init__(*args, **kwargs)


class QuerySet(models.QuerySet):
    def as_dataframe(self, *fields, natural=False):
        """
        Convert to pandas dataframe

        :param: fields str: Only return columns with given field names.
        :param: natural bool: If true, then replace id/pk of foreign
                              relation with natural representation.
        """
        index=self.values_list('id', flat=True)
        df = pandas.DataFrame([], index=index)
        for i in self.model._meta.get_fields():
            if fields and i.name not in fields:
                continue
            if not Model.is_simple_field(i):
                continue
            if i.name == 'id':
                continue

            dtype = Model.pd_type(i)
            if i.is_relation and natural:
                col_dat = map(
                    lambda obj:
                        getattr(getattr(obj, i.name), 'natural', None),
                    self.prefetch_related()
                )
            else:
                col_dat = self.values_list(i.name, flat=True)
            kwargs = dict(index=index)
            if i.choices:
                col_dat = pandas.Categorical(col_dat)
            else:
                if dtype is str:
                    # None become empty str
                    # prevents 'None' string to enter df str columns
                    col_dat = ('' if i is None else i for i in col_dat)
                kwargs['dtype'] = dtype
            df[i.name] = pandas.Series(col_dat, **kwargs)
        return df

    def _filter_or_exclude(self, negate, *args, **kwargs):
        """
        Handle natural lookups for filtering operations
        """
        kwargs = self.model.handle_natural_lookups(**kwargs)
        return super()._filter_or_exclude(negate, *args, **kwargs)

    def _values(self, *fields, **expressions):
        """
        Handle the 'natural' fields for value retrievals
        """
        if 'natural' in fields:
            fields = [i for i in fields if i != 'natural']
        return super()._values(*fields, **expressions)

    def get_field_stats(self, fieldname, natural=False):
        """
        Get basic descriptive stats from a single field/column

        Returns a dict: stats_type -> obj
        Returning an empty dict indicates some error
        """
        if fieldname == 'id':
            # as_dataframe('id') does not return anything meaningful
            # FIXME?
            return {}

        qs = self
        if natural and self.model._meta.get_field(fieldname).is_relation:
            # speedup
            qs = qs.select_related(fieldname)

        try:
            col = qs.as_dataframe(fieldname, natural=natural)[fieldname]
        except Exception as e:
            # as_dataframe may not support some data types
            log.debug('get_field_stats() failed:', type(e), e)
            raise
            return {}

        count_stats = col.value_counts(dropna=False).sort_index()

        ret = {
            'choice_counts': count_stats,
            'description': col.describe(),
        }

        if count_stats.count() == 1:
            # all values the same
            ret['uniform'] = count_stats.to_dict()

        if count_stats.max() < 2:
            # column values are unique
            ret['unique'] = count_stats.max()

        try:
            not_blank = count_stats.drop(index='')
        except KeyError:
            pass
        else:
            if not_blank.max() < 2:
                # column unique except for empties
                ret['unique_blank'] = {
                    'BLANK': count_stats[''],
                    'NOT_BLANK': not_blank.sum(),
                }

        return ret


class Manager(models.Manager):
    def get_queryset(self):
        return QuerySet(self.model, using=self._db)

    def get_by_natural_key(self, key):
        return self.get(**self.model.natural_lookup(key))


class PublishManager(Manager):
    """
    Manager to implement publishable vs. hidden data
    """
    filter = None
    excludes = None

    def __init__(self, *args, filter={}, excludes=[], **kwargs):
        super().__init__(*args, **kwargs)
        # Setting base filters, must be completed after all the managers are
        # fully initialized but that won't happen until the models are set up
        self.base_filter = filter.copy()
        self.base_excludes = []
        for i in excludes:
            self.base_excludes.append(i.copy())

    def ensure_filter_setup(self):
        """
        Follow foreign key relations recursively to set up filters

        TODO: handle cycles
        """
        if self.filter is not None and self.excludes is not None:
            return

        filter = self.base_filter.copy()
        excludes = [i.copy() for i in self.base_excludes]

        for i in self.model.get_fields().fields:
            if i.is_relation and i.many_to_one:
                # is a foreign key
                other = i.related_model.published
                prefix = i.name + '__'
                other.ensure_filter_setup()
                # FIXME: infinite recursion is waiting to happen
                for k, v in other.filter.items():
                    filter[prefix + k] = v
                for i in other.excludes:
                    e = {prefix + k: v for k, v in i.items()}
                    if e:
                        excludes.append(e)

        self.filter = filter
        self.excludes = excludes

    def get_queryset(self):
        self.ensure_filter_setup()
        qs = super().get_queryset()

        if self.filter is not None:
            qs = qs.filter(**self.filter)
        for i in self.excludes:
            qs = qs.exclude(**i)

        return qs

    def get_publish_filter(self):
        """
        Return the publish filter/exclude as Q object

        A Q return value can be used for the filter keyword in calls to
        Aggregate() and friends, e.g. Count().  A None return value can be used
        to determine that no such filter needs tpo be applied.

        The model name will be added to the lookup lhs.
        """
        self.ensure_filter_setup()
        prefix = self.model._meta.model_name + '__'
        f = {prefix + k: v for k, v in self.filter.items()}
        e = [
            ~Q(**{prefix + k: v for k, v in i.items()})
            for i in self.excludes if i
        ]
        q = Q(*e, **f)
        if q:
            return q
        else:
            return None


Fields = namedtuple('Fields', ['fields', 'names', 'verbose'])
""" container to hold list of fields for a model """


class AutoField(models.AutoField):
    """
    An AutoField with a verbose name that includes the model
    """
    def contribute_to_class(self, cls, name, **kwargs):
        # at super() this sets self.model, so we can patch the verbose name
        # after this
        super().contribute_to_class(cls, name, **kwargs)
        self.verbose_name = self.model._meta.model_name + '_' + self.name


class ChangeRecord(models.Model):
    """
    Model representing a changelog entry
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    file = models.FileField(upload_to='import/%Y/', null=True,
                            verbose_name='data source file')
    line = models.IntegerField(
        null=True, help_text='The corresponding line in the input file',
    )
    command_line = models.CharField(
        max_length=200, blank=True, help_text='management command for import')
    record_type = models.ForeignKey(ContentType, on_delete=models.PROTECT)
    record_pk = models.PositiveIntegerField()
    record = GenericForeignKey('record_type', 'record_pk')
    record_natural = models.CharField(max_length=300, blank=True)
    fields = models.TextField()
    is_created = models.BooleanField(default=False, verbose_name='new record')
    is_deleted = models.BooleanField(default=False)

    class Meta:
        #app_label = 'mibios'
        get_latest_by = 'timestamp'
        ordering = ['-timestamp']

    def __str__(self):
        user = ' ' + self.user.username if self.user else ''

        if self.is_deleted:
            return '{}{} (deleted) - {} (pk:{})'.format(
                self.timestamp, user, self.record_natural, self.record_pk
            )

        new = ' (new)' if self.is_created else ''
        dots = '...' if len(self.fields) > 80 else ''
        return '{}{}{} - {}{}'.format(self.timestamp, user, new,
                                      self.fields[:80], dots)

    def has_changed(self):
        """
        Compare with previous change record of same object
        """
        qs = self.record.history
        if self.timestamp is not None:
            qs = qs.filter(timestamp__lt=self.timestamp)

        try:
            prev = qs.latest()
        except self.DoesNotExist:
            # we are first
            return True

        if self.record_natural != prev.record_natural:
            return True

        if not self.fields:
            self.serialize()

        if self.fields != prev.fields:
            return True

        return False

    def serialize(self):
        """
        Serialize field content
        """
        if self.is_deleted:
            return

        self.fields = serializers.serialize(
            'json',
            [self.record],
            fields=self.record.get_fields(skip_auto=True, with_m2m=True).names,
            use_natural_foreign_keys=True
        )


class Model(models.Model):
    """
    Adds some extras to Django's Model
    """

    """
    String value, indicating missing data to be used externally.  Internally,
    None or the empty string remains in use for missing data.
    """
    MISSING_DATA = '-'

    # replace the default auto field that Django adds
    id = AutoField(primary_key=True)
    history = models.ManyToManyField(ChangeRecord)

    class Meta:
        abstract = True

    objects = Manager()
    published = PublishManager()

    @classmethod
    def pd_type(cls, field):
        """
        Map Django field type to pandas data type
        """
        str_fields = (
            models.CharField,
            models.TextField,
            models.ForeignKey,
        )
        int_fields = (
            models.IntegerField,
            models.AutoField,
        )
        if isinstance(field, str_fields):
            dtype = str
        elif isinstance(field, int_fields):
            dtype = pandas.Int64Dtype()
        elif isinstance(field, models.BooleanField):
            dtype = bool
        elif isinstance(field, models.DecimalField):
            dtype = float
        else:
            raise ValueError('Field type not supported: {}: {}'
                             ''.format(field, field.get_internal_type()))
        return dtype

    @classmethod
    def is_simple_field(cls, field):
        """
        Check if given field is "simple"

        Simple fields are not ManyToMany or ManyToOne but can be represented
        in a table cell
        """
        if isinstance(field, (models.ManyToOneRel, models.ManyToManyField)):
            return False
        else:
            return True

    @classmethod
    def get_fields(cls, skip_auto=False, with_m2m=False):
        """
        Get fields to be displayed in table (in order) and used for import

        Should be overwritten by models as needed to include e.g. m2m fields
        Many-to-many fields are by default excluded because of the difficulties
        of meaningfully displaying them
        """
        exclude = [
            models.ManyToOneRel,
        ]
        if skip_auto:
            exclude.append(models.AutoField)

        if not with_m2m:
            exclude.append(models.ManyToManyRel)
            exclude.append(models.ManyToManyField)

        exclude = tuple(exclude)

        fields = [
            i for i
            in cls._meta.get_fields()
            if not isinstance(i, exclude) and i.name != 'history'
        ]
        names = [i.name for i in fields]
        verbose = [getattr(i, 'verbose_name', i.name) for i in fields]
        return Fields(fields=fields, names=names, verbose=verbose)

    def export(self):
        """
        Convert object into "table row" / list
        """
        ret = []
        for i in self._meta.get_fields():
            if self.is_simple_field(i):
                ret.append(getattr(self, i.name, None))
        return ret

    def export_dict(self, to_many=False):
        ret = OrderedDict()
        for i in self._meta.get_fields():
            if self.is_simple_field(i) or to_many:
                ret[i.name] = getattr(self, i.name, None)
        return ret

    def compare(self, other):
        """
        Compares two objects and relates them by field content

        Can be used to determine if <self> can be updated by <other> in a
        purely additive, i.e. without changing existing data, just filling
        blank fields.  <other> can also be a dict.

        Returns a tuple (bool, int), the first component of which says if both
        objects are consistent with each other, i.e. if the only differences on
        fields involve one of the fields being blank or null.  Differences on
        many-to-many fields don't affect consistency.  The second component
        contains the names of those fields that are null or blank in <self> but
        not in <other> including additional many-to-many links.

        For two inconsistent objects the return value's second component is
        undefined (it may be usable for debugging.)
        """
        if isinstance(other, Model):
            if self._meta.concrete_model != other._meta.concrete_model:
                return (False, None)
        elif not isinstance(other, dict):
            raise TypeError('can\'t compare to {} object'.format(type(other)))

        diff = []
        is_consistent = True
        non_matching = []
        for i in self._meta.get_fields():
            if isinstance(other, dict) and i.name not in other:
                # interpret as blank/None in other (dict version)
                continue

            if isinstance(i, models.ManyToOneRel):
                # a ForeignKey in third model pointing to us
                # ignore - must be handled from third model
                pass
            elif isinstance(i, models.ManyToManyField):
                ours = set(getattr(self, i.name).all())
                if isinstance(other, dict):
                    # TODO / FIXME: how do we get here?
                    try:
                        theirs = set(other['name'])
                    except TypeError:
                        # packaged in iterable for set()
                        theirs = set([other['name']])
                    except KeyError:
                        theirs = set()
                else:
                    theirs = set(getattr(other, i.name).all())
                if theirs - ours:
                    diff.append(i.name)
            elif isinstance(i, models.OneToOneField):
                raise NotImplementedError()
            else:
                # ForeignKey or normal scalar field
                # Assumes that None and '' are not both possible values and
                # that either of them indicates missing data
                ours = getattr(self, i.name)
                if isinstance(other, dict):
                    theirs = other[i.name]
                else:
                    theirs = getattr(other, i.name)

                if theirs is None or theirs == '':
                    # other data missing ok
                    continue
                elif ours is None or ours == '':
                    # other has more data
                    diff.append(i.name)
                    continue

                # both are real data
                # usually other dict has str values
                # try to cast to e.g. Decimal, ... (crossing fingers?)
                if isinstance(theirs, str):
                    theirs = type(ours)(theirs)

                if ours != theirs:
                    is_consistent = False
                    non_matching.append(i.name)

        return (is_consistent, diff if is_consistent else non_matching)

    @property
    def natural(self):
        """
        A natural identifier under which the object is commonly known

        This defaults to the name field if it exists.  Models without a name
        field should implement this.

        The natural value must be derived from the non-relational fields of
        the model.  To implement the natural proterty for a model this method
        as well as natural_lookup() must be implemented.  The setter method
        should be general enough for most cases.  The inverse of this method is
        implemented by natural_lookup().
        """
        return getattr(self, 'name', self.pk)

    def natural_key(self):
        return self.natural

    @classmethod
    def natural_lookup(cls, value):
        """
        Generate a dict lookup from the natural value / key

        Used to replace a natural lookup with real lookups that Django can
        understand.  This method should be overwritten to suit the model.  The
        default implementation assumes the model has a "name" field.
        Implementations usually de-construct the natural value into its
        components and is the inverse of the natural() property.

        Might also be used as parsing tool to coerce a user-given input value
        to field-compatible values.
        """
        try:
            cls._meta.get_field('name')
        except FieldDoesNotExist:
            return dict(pk=value)
        else:
            return dict(name=value)

    @natural.setter
    def natural(self, value):
        """
        Update model fields from natural value

        The default implementation should be general enough to be used by
        inheriting classes
        """
        for k, v in self.natural_lookup(value).items():
            setattr(self, k, v)

    def __str__(self):
        return self.natural

    @classmethod
    def handle_natural_lookups(cls, **lookups):
        """
        Detect and convert natural object lookups

        Convert "*__natural='foo'" and "*__model_name='foo'" into their
        proper lookups
        """
        ret = {}
        for lhs, rhs in lookups.items():
            if not isinstance(rhs, (str, int)):
                # any natural lookup should be str (or int for pk), if it's
                # not, then the resulting error will be probably be misleading
                ret[lhs] = rhs
                continue

            cur_model = cls

            parts = lhs.split('__')
            for part in parts:
                if part == 'natural':
                    continue
                fields = {i.name: i for i in cur_model._meta.get_fields()}
                if part in fields and fields[part].is_relation:
                    cur_model = fields[part].related_model
                else:
                    # not an obj lookup, keep as-is
                    ret[lhs] = rhs
                    break
            else:
                # prepare lhs prefix for the natural replacements:
                if part == 'natural':
                    # remove __natural from lhs
                    lhs = '__'.join(parts[:-1])

                if lhs:
                    lhs += '__'

                if isinstance(rhs, int):
                    ret.update({lhs + 'pk': rhs})
                else:
                    try:
                        real_lookups = cur_model.natural_lookup(rhs)
                    except Exception as e:
                        # Assume code in natural_lookup() is correct and
                        # treat this a user error, i.e. the natural rhs is
                        # bad
                        msg = 'Failed to resolve: [{}]{}={}' \
                              ''.format(cur_model._meta.model_name, parts, rhs)
                        raise NaturalKeyLookupError(msg) from e

                    ret.update({lhs + k: v for k, v in real_lookups.items()})
        return ret

    @classmethod
    def str_blank(cls, *values):
        """
        Convert into strings, explicitly marking blank/null values as such

        Use this when an empty string is insufficient to indicate missing data.
        This is the reverse of decode_blank().
        """
        ret = []
        for i in values:
            if i in ['', None]:
                ret.append(cls.MISSING_DATA)
            else:
                ret.append(str(i))
        return ret[0] if len(ret) == 1 else tuple(ret)

    @classmethod
    def decode_blank(cls, *values):
        """
        Filter values, turning strings indicating missing data into actual
        blank/empty string.
        """
        ret = []
        for i in values:
            if i == cls.MISSING_DATA:
                ret.append('')
            else:
                ret.append(i)
        return ret[0] if len(ret) == 1 else tuple(ret)

    def get_absolute_url(self):
        name = 'mibios:{app}_{model}_change' \
               ''.format(app=self._meta.app_label, model=self._meta.model_name)
        return reverse(name, kwargs=dict(object_id=self.pk))

    def add_change_record(self, is_deleted=False, file=None, line=None,
                          user=None, cmdline=''):
        """
        Create a change record attribute for this object

        If the object has no id/pk yet the change will be "is_created".  The
        fields will remain empty until save()
        """
        self.change = ChangeRecord(
            user=user,
            file=file,
            line=line,
            command_line=cmdline,
            record=self,
            record_natural=self.natural,
            is_created=self.id is None,
            is_deleted=is_deleted,
        )

    @transaction.atomic
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if not hasattr(self, 'change'):
            self.add_change_record()
        # set record (again) as super().save() resets this to None for unknown
        # reasons:
        self.change.record = self
        self.change.serialize()
        if self.change.has_changed():
            self.change.save()
            self.history.add(self.change)
        del self.change

    def delete(self, *args, **kwargs):
        if not hasattr(self, 'change'):
            self.add_change_record(is_delete=True)
        self.change.serialize()
        with transaction.atomic():
            self.change.save()
            super().delete(*args, **kwargs)

    def full_clean(self, *args, **kwargs):
        """
        Validate the object

        Add model name to super()'s error dict
        """
        try:
            super().full_clean(*args, **kwargs)
        except ValidationError as e:
            errors = e.update_error_dict({
                'model_name': self._meta.model_name,
            })
            raise ValidationError(errors)


# utility functions
def erase_all_data(verbose=False):
    """
    Delete all data
    """
    if verbose:
        print('Erasing all data...', file=sys.stderr)
    for m in apps.get_app_config('mibios').get_models():
        m.objects.all().delete()


def show_stats():
    """
    print db stats
    """
    for m in apps.get_app_config('mibios').get_models():
        print('{}: {}'.format(m._meta.label, m.objects.count()))
