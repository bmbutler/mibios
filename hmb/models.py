from collections import OrderedDict
import re
import sys

from django.apps import apps
from django.core.exceptions import FieldDoesNotExist
from django.db import models
import pandas


class QuerySet(models.QuerySet):
    def as_dataframe(self):
        """
        Convert to pandas dataframe
        """
        df = pandas.DataFrame([], index=self.values_list('id', flat=True))
        for i in self.model._meta.get_fields():
            if not Model.is_simple_field(i):
                continue
            if i.name == 'id':
                continue

            dtype = Model.pd_type(i)
            col_dat = self.values_list(i.name, flat=True)
            kwargs = dict()
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

    def _filter_or_exclude(self, negate, *args, canonical=None, **kwargs):
        """
        Implement canonical lookups
        """
        c = {}
        if canonical is not None:
            c = self.model.canonical_lookup(canonical)
        return super()._filter_or_exclude(negate, *args, **c, **kwargs)


class Manager(models.Manager):
    def get_queryset(self):
        return QuerySet(self.model, using=self._db)


class Model(models.Model):
    class Meta:
        abstract = True

    objects = Manager()

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
    def get_simple_fields(cls):
        """
        Get forward, non-many-to-fields, non-auto
        """
        non_simple = (
            models.AutoField,
            models.ManyToOneRel,
            models.ManyToManyField,
        )
        return [
            i for i
            in cls._meta.get_fields()
            if not isinstance(i, non_simple)
        ]

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
                    try:
                        theirs = set(other['name'])
                    except TypeError:
                        # packaged in iterable for set()
                        theirs = set([other['name']])
                else:
                    theirs = set(getattr(other, i.name).all())
                if theirs - ours:
                    diff.append(i.name)
            elif isinstance(i, models.OneToOneField):
                raise NotImplementedError()
            else:
                # ForeignKey or normal scalar field
                # Assumes that None and '' are not both possible values and
                # that one of them indicates missing data
                ours = getattr(self, i.name)
                if isinstance(other, dict):
                    # expect other dict to have str values
                    # try to cast to e.g. Decimal, ... (crossing fingers?)
                    theirs = type(ours)(other[i.name])
                else:
                    theirs = getattr(other, i.name)
                if ours == theirs:
                    pass
                elif ours is None or ours == '':
                    diff.append(i.name)
                elif theirs is None or theirs == '':
                    pass
                else:
                    # both sides differ with values
                    is_consistent = False
                    non_matching.append(i.name)

        return (is_consistent, diff if is_consistent else non_matching)

    @property
    def canonical(self):
        """
        A canonical identifier under which the object is commonly known

        This defaults to the name field if it exists.  Models without a name
        field should implement this.

        The canonical value must be derived from the non-relational fields of
        the model.  To implement the canonical proterty for a model this method
        as well as canonical_lookup() must be implemented.  The setter method
        should be general enough for most cases.  The inverse of this method is
        implemented by canonical_lookup().
        """
        return getattr(self, 'name', self.pk)

    @classmethod
    def canonical_lookup(cls, value):
        """
        Generate a dict lookup from the canonical value

        This is used in QuerySet.get() to find objects canonical value.  The
        implementation usually de-constructs the canonical value into its
        components and is the inverse of the canonical() property.

        Might also be used as parsing tool to coerce a user-given input value
        to field-compatible values.
        """
        try:
            cls._meta.get_field('name')
        except FieldDoesNotExist:
            return dict(pk=value)
        else:
            return dict(name=value)

    @canonical.setter
    def canonical(self, value):
        """
        Update model fields from canonical value

        The default implementation should be general enough to be used by
        inheriting classes
        """
        for k, v in self.canonical_lookup(value).items():
            setattr(self, k, v)

    def __str__(self):
        return self.canonical


class Diet(Model):
    # FIXME: two below are suggested by Tom's diagram:
    #   composition = models.CharField(max_length=1000)
    #   week = models.ForeignKey('Week', on_delete=models.CASCADE)

    # TODO: these all should have choices
    frequency = models.CharField(max_length=30, blank=True)
    dose = models.DecimalField(
        max_digits=4, decimal_places=1,
        blank=True, null=True, verbose_name='total dose grams'
    )
    supplement = models.CharField(
        max_length=200, blank=True, verbose_name='supplement consumed'
    )

    class Meta:
        unique_together = (
            ('frequency', 'dose', 'supplement'),
        )
        ordering = ('supplement', 'frequency', 'dose')

    @property
    def canonical(self):
        return '{} {} {}'.format(self.supplement, self.frequency, self.dose)

    @classmethod
    def canonical_lookup(cls, value):
        s, f, d = value.split()
        return dict(supplement=s, frequency=f, dose=d)


class FecalSample(Model):
    participant = models.ForeignKey('Participant', on_delete=models.CASCADE)
    number = models.PositiveSmallIntegerField()
    week = models.ForeignKey('Week', on_delete=models.SET_NULL, blank=True,
                             null=True)
    ph = models.DecimalField(max_digits=4, decimal_places=2, blank=True,
                             null=True)
    bristol = models.DecimalField(max_digits=3, decimal_places=1, blank=True,
                                  null=True)
    note = models.ManyToManyField('Note')
    # SCFA stuff
    # relatives seem to be calculated with lots of digits
    scfa_kw = dict(max_digits=20, decimal_places=14, blank=True, null=True)
    final_weight = models.DecimalField(**scfa_kw)
    acetate_abs = models.DecimalField(**scfa_kw, verbose_name='Acetate_mM')
    acetate_rel = models.DecimalField(**scfa_kw,
                                      verbose_name='Acetate_mmol_kg')
    butyrate_abs = models.DecimalField(**scfa_kw,
                                       verbose_name='Butyrate_mM')
    butyrate_rel = models.DecimalField(**scfa_kw,
                                       verbose_name='Butyrate_mmol_kg')
    propionate_abs = models.DecimalField(**scfa_kw,
                                         verbose_name='Propionate_mM')
    propionate_rel = models.DecimalField(**scfa_kw,
                                         verbose_name='Propionate_mmol_kg')

    class Meta:
        unique_together = ('participant', 'number')
        ordering = ('participant', 'number')

    id_pat = re.compile(r'^(?P<participant>(NP|U)[0-9]+)_(?P<num>[0-9]+)$')

    @classmethod
    def parse_id(cls, txt):
        """
        Convert sample identifing str into kwargs dict

        """
        # FIXME: returns participant as str, but compare to canonical_lookup
        # which returns a model object
        m = cls.id_pat.match(txt.strip())
        if m is None:
            raise ValueError('Failed parsing sample id: {}'.format(txt[:100]))
        else:
            m = m.groupdict()
            participant = m['participant']
            number = m['num']

        number = int(number)

        return {'participant': participant, 'number': number}

    @property
    def name(self):
        """
        Allow us to display the canonical value as a name (in tables)
        """
        return self.canonical

    @property
    def canonical(self):
        return '{}_{}'.format(self.participant, self.number)

    @classmethod
    def canonical_lookup(cls, value):
        p, n = value.split('_')
        p = Participant.object.get(name=p)
        return dict(participant=p, number=int(n))


class Note(Model):
    name = models.CharField(max_length=100, unique=True)
    text = models.TextField(max_length=5000)


class Participant(Model):
    name = models.CharField(max_length=50, unique=True)
    sex = models.CharField(max_length=50, blank=True)
    age = models.SmallIntegerField(blank=True, null=True)
    ethnicity = models.CharField(max_length=200, blank=True)
    semester = models.ForeignKey('Semester', on_delete=models.CASCADE,
                                 blank=True, null=True)
    diet = models.ForeignKey('Diet', on_delete=models.SET_NULL, blank=True,
                             null=True)
    QUANTITY_COMPLIANCE_CHOICES = ['NA', 'Quantity_compliant', 'no', 'none',
                                   'unknown', 'yes']
    _qc_choices = [(i, i) for i in QUANTITY_COMPLIANCE_CHOICES]

    quantity_compliant = models.CharField(
        max_length=30, choices=_qc_choices, blank=True,
        help_text='Did the participant consumed at least 75% of the starch '
                  'they were prescribed?'
    )
    note = models.ManyToManyField('Note')

    class Meta:
        ordering = ['semester', 'name']


class Semester(Model):
    # semester: 4 terms, numeric, so they can be sorted, winter goes first
    WINTER = '1'
    FALL = '4'
    TERM_CHOICES = (
        (FALL, 'fall'),
        (WINTER, 'winter'),
    )
    term = models.CharField(max_length=20, choices=TERM_CHOICES)
    year = models.PositiveSmallIntegerField()

    class Meta:
        unique_together = ('term', 'year')
        # FIXME: unsure why -term makes winter come before fall as 1<4
        ordering = ['year', '-term']

    @property
    def canonical(self):
        return self.term.capitalize() + str(self.year)

    pat = re.compile(r'^(?P<term>[a-zA-Z]+)[^a-zA-Z0-9]*(?P<year>\d+)$')

    @classmethod
    def canonical_lookup(cls, txt):
        m = cls.pat.match(txt.strip())
        if m is None:
            raise ValueError('Failed parsing as semester: {}'.format(txt[:99]))
        else:
            term, year = m.groups()

        term = term.lower()
        year = int(year)
        if year < 100:
            # two-digit year given, assume 21st century
            year += 2000

        return dict(term=term, year=year)


class Sequencing(Model):
    MOCK = 'mock'
    WATER = 'water'
    BLANK = 'blank'
    PLATE = 'plate'
    OTHER = 'other'
    CONTROL_CHOICES = (
        (MOCK, MOCK),
        (WATER, WATER),
        (BLANK, BLANK),
        (PLATE, PLATE),
        (OTHER, OTHER),
    )
    name = models.CharField(max_length=100, unique=True)
    sample = models.ForeignKey('FecalSample', on_delete=models.CASCADE,
                               blank=True, null=True)
    control = models.CharField(max_length=50, choices=CONTROL_CHOICES,
                               blank=True)
    r1_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    r2_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    note = models.ManyToManyField('Note')
    run = models.ForeignKey('SequencingRun', on_delete=models.CASCADE,
                            blank=True, null=True)
    plate = models.PositiveSmallIntegerField(blank=True, null=True)
    plate_position = models.CharField(max_length=10, blank=True)
    snumber = models.PositiveSmallIntegerField(blank=True, null=True)

    class Meta:
        unique_together = (
            ('run', 'snumber'),
            ('run', 'plate', 'plate_position'),
        )
        ordering = ['name']

    @classmethod
    def parse_control(cls, txt):
        """
        Coerce text into available control choices
        """
        choice = txt.strip().lower()
        if choice:
            for i in (j[0] for j in cls.CONTROL_CHOICES):
                if i in choice:
                    return i
            return cls.OTHER
        else:
            return ''


class SequencingRun(Model):
    serial = models.CharField(max_length=50)
    number = models.PositiveSmallIntegerField()
    path = models.CharField(max_length=2000, blank=True)

    class Meta:
        unique_together = ('serial', 'number')
        ordering = ['serial', 'number']

    @property
    def canonical(self):
        return '{}-{}'.format(self.serial, self.number)

    @classmethod
    def canonical_lookup(cls, value):
        s, n = value.split('-')
        return dict(serial=s, number=int(n))


class Week(Model):
    number = models.PositiveSmallIntegerField(unique=True)

    class Meta:
        ordering = ('number',)

    @property
    def canonical(self):
        return 'week{}'.format(self.number)

    pat = re.compile(r'(week[^a-zA-Z0-9]*)?(?P<num>[0-9]+)', re.IGNORECASE)

    @classmethod
    def canonical_lookup(cls, value):
        """
        Convert a input text like "Week 1" into {'number' : 1}
        """
        m = cls.pat.match(value)
        if m is None:
            raise ValueError(
                'Failed to parse this as a week: {}'.format(value[:100])
            )
        return dict(number=int(m.groupdict()['num']))


class Community(Model):
    asv = models.ManyToManyField('ASV')
    seqs = models.ForeignKey('Sequencing', on_delete=models.CASCADE)


class Strain(Model):
    asv = models.ForeignKey('ASV', on_delete=models.SET_NULL, blank=True,
                            null=True)


class BreathSample(Model):
    participant = models.ForeignKey('Participant', on_delete=models.CASCADE)
    week = models.ForeignKey('Week', on_delete=models.SET_NULL, blank=True,
                             null=True)
    gases = models.CharField(max_length=100)


class BloodSample(Model):
    participant = models.ForeignKey('Participant', on_delete=models.CASCADE)
    week = models.ForeignKey('Week', on_delete=models.SET_NULL, blank=True,
                             null=True)
    cytokines = models.CharField(max_length=100)


class ASV(Model):
    number = models.PositiveIntegerField()
    taxon = models.ForeignKey('Taxon', on_delete=models.SET_NULL, blank=True,
                              null=True)


class Taxon(Model):
    taxid = models.PositiveIntegerField()
    organism = models.CharField(max_length=100)


# utility functions


def erase_all_data(verbose=False):
    """
    Delete all data
    """
    if verbose:
        print('Erasing all data...', file=sys.stderr)
    for m in apps.get_app_config('hmb').get_models():
        m.objects.all().delete()


def show_stats():
    """
    print db stats
    """
    for m in apps.get_app_config('hmb').get_models():
        print('{}: {}'.format(m._meta.label, m.objects.count()))
