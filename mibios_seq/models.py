from django.apps import apps
from django.db import models

from . import __version__
from mibios.dataset import registry
from mibios.models import Model
from mibios.utils import getLogger

from hhcd.models import FecalSample, Note


log = getLogger(__name__)


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
    sample = models.ForeignKey(FecalSample, on_delete=models.CASCADE,
                               blank=True, null=True)
    control = models.CharField(max_length=50, choices=CONTROL_CHOICES,
                               blank=True)
    r1_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    r2_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    note = models.ManyToManyField(Note, blank=True)
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

    @Model.natural.getter
    def natural(self):
        return '{}-{}'.format(self.serial, self.number)

    @classmethod
    def natural_lookup(cls, value):
        s, n = value.split('-')
        return dict(serial=s, number=int(n))


class Community(Model):
    asv = models.ManyToManyField('ASV')
    seqs = models.ForeignKey('Sequencing', on_delete=models.CASCADE)


class Strain(Model):
    asv = models.ForeignKey('ASV', on_delete=models.SET_NULL, blank=True,
                            null=True)


class ASV(Model):
    number = models.PositiveIntegerField()
    taxon = models.ForeignKey('Taxon', on_delete=models.SET_NULL, blank=True,
                              null=True)


class Taxon(Model):
    taxid = models.PositiveIntegerField()
    organism = models.CharField(max_length=100)


registry.add_models()
registry.apps[apps.get_app_config('mibios_seq').name] = dict(
    version=__version__,
)
