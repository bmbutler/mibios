from collections import OrderedDict
import re

from django.apps import apps
from django.db import models
import pandas


class QuerySet(models.QuerySet):
    def as_dataframe(self):
        """
        Convert to pandas dataframe
        """
        data = OrderedDict(
            [(i.name, []) for i in self.model._meta.get_fields()]
        )
        for i in self:
            d = i.export_dict()

            for k, v in d.items():
                data[k].append(v)

        df = pandas.DataFrame(data, columns=data.keys())
        return df


class Manager(models.Manager):
    def get_queryset(self):
        return QuerySet(self.model, using=self._db)


class Model(models.Model):
    class Meta:
        abstract = True

    objects = Manager()

    def export(self):
        """
        Convert object into "table row" / list
        """
        ret = []
        for i in self._meta.get_fields():
            ret.append(getattr(self, i.name, None))
        return ret

    def export_dict(self):
        ret = OrderedDict()
        for i in self._meta.get_fields():
            ret[i.name] = getattr(self, i.name, None)
        return ret


class Diet(Model):
    composition = models.CharField(max_length=1000)
    week = models.ForeignKey('Week', on_delete=models.CASCADE)


class FecalSample(Model):
    participant = models.ForeignKey('Participant', on_delete=models.CASCADE)
    number = models.PositiveSmallIntegerField()
    week = models.ForeignKey('Week', on_delete=models.SET_NULL, blank=True,
                             null=True)
    note = models.ManyToManyField('Note')

    class Meta:
        unique_together = ('participant', 'number')
        ordering = ('participant', 'number')

    id_pat = re.compile(r'^(?P<participant>(NP|U)[0-9]+)_(?P<num>[0-9]+)$')

    @classmethod
    def parse_id(cls, txt):
        """
        Convert sample identifing str into kwargs dict
        """
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
        return '{}_{}'.format(self.participant, self.number)

    def __str__(self):
        return self.name


class Note(Model):
    name = models.CharField(max_length=100, unique=True)
    text = models.TextField(max_length=300)

    def __str__(self):
        return self.name


class Participant(Model):
    name = models.CharField(max_length=50, unique=True)
    sex = models.CharField(max_length=50, blank=True)
    age = models.SmallIntegerField(blank=True, null=True)
    ethnicity = models.CharField(max_length=200, blank=True)
    semester = models.ForeignKey('Semester', on_delete=models.CASCADE,
                                 blank=True, null=True)
    diet = models.ForeignKey('Diet', on_delete=models.SET_NULL, blank=True,
                             null=True)
    note = models.ManyToManyField('Note')

    class Meta:
        ordering = ['semester', 'name']

    def __str__(self):
        return self.name


class Semester(Model):
    # semester: 4 seasons, numeric, so they can be sorted
    FALL = '3'
    WINTER = '4'
    SEASON_CHOICES = (
        (FALL, 'fall'),
        (WINTER, 'winter'),
    )
    season = models.CharField(max_length=20, choices=SEASON_CHOICES)
    year = models.PositiveSmallIntegerField()

    class Meta:
        unique_together = ('season', 'year')
        ordering = ['year', 'season']

    def __str__(self):
        return self.season.capitalize() + str(self.year)

    pat = re.compile(r'^(?P<season>[a-zA-Z]+)[^a-zA-Z0-9]*(?P<year>\d+)$')

    @classmethod
    def parse(cls, txt):
        """
        Convert str into kwargs dict
        """
        m = cls.pat.match(txt.strip())
        if m is None:
            raise ValueError('Failed parsing as semester: {}'.format(txt[:99]))
        else:
            season, year = m.groups()

        season = season.lower()
        year = int(year)
        if year < 100:
            # two-digit year given, assume 21st century
            year += 2000

        return {'season': season, 'year': year}


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
    name = models.CharField(max_length=100)
    sample = models.ForeignKey('FecalSample', on_delete=models.CASCADE,
                               blank=True, null=True)
    control = models.CharField(max_length=50, choices=CONTROL_CHOICES,
                               blank=True)
    r1_file = models.CharField(max_length=100, unique=True, blank=True,
                               null=True)
    r2_file = models.CharField(max_length=100, unique=True, blank=True,
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
        ordering = ['run__serial', 'run__number', 'name']

    def __str__(self):
        return self.name


class SequencingRun(Model):
    serial = models.CharField(max_length=50)
    number = models.PositiveSmallIntegerField()
    path = models.CharField(max_length=2000, blank=True)

    class Meta:
        unique_together = ('serial', 'number')
        ordering = ['serial', 'number']

    def __str__(self):
        return '{} {}'.format(self.serial, self.number)


class Week(Model):
    number = models.PositiveSmallIntegerField(unique=True)

    class Meta:
        ordering = ('number',)

    def __str__(self):
        return 'week{}'.format(self.number)

    pat = re.compile(r'(week[^a-zA-Z0-9]*)?(?P<num>[0-9]+)', re.IGNORECASE)

    @classmethod
    def parse(cls, txt):
        """
        Convert a input text like "Week 1" into {'number' : 1}
        """
        m = cls.pat.match(txt)
        if m is None:
            raise ValueError(
                'Failed to parse this as a week: {}'.format(txt[:100])
            )
        return {'number': int(m.groupdict()['num'])}


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


def erase_all_data():
    """
    Delete all data
    """
    for m in apps.get_app_config('hmb').get_models():
        m.objects.all().delete()


def show_stats():
    """
    print db stats
    """
    for m in apps.get_app_config('hmb').get_models():
        print('{}: {}'.format(m._meta.label, m.objects.count()))
