""" parking space for distance-related modeling """
from django.db import models

from mibios.models import Model


class Distance(Model):
    """
    A (partial) distance matrix

    source.accession sorts strictly before target.accession

    To implement this model, two foreign keys (targeting the same model) need
    to be added in the inheriting class: source and target
    """
    DEFAULT_METHOD = ''

    source = None  # provided by implementing class
    target = None  # provided by implementing class
    method = models.CharField(
        max_length=32, blank=True, default=DEFAULT_METHOD,
    )
    value = models.FloatField()
    history = None

    class Meta:
        abstract = True
        unique_together = (
            ('source', 'target', 'method',),
        )

    def __str__(self):
        return f'{self.source.accession}-{self.target.accession}:{self.value}'

    @classmethod
    def from_tuples(cls, data, method=DEFAULT_METHOD):
        """
        populated tables from triplet of (src, tgt, distance)

        Sources and targets need to be model instances.
        """
        data = (
            (*((a, b) if a.name < b.name else (b, a)), v)
            for a, b, v in data
        )
        objs = (
            cls(source=s, target=t, value=v, method=method)
            for s, t, v in data
        )
        # TODO: see docs on bulk_create() how to avoid generator-to-list
        # casting
        return cls.objects.bulk_create(objs)

    @classmethod
    def validate_order(cls):
        """
        Checks integrity of source-target sorting contraint
        """
        qs = cls.objects.values_list('source__accession', 'target__accession')
        for i, j in qs.iterator():
            if i >= j:
                raise ValidationError(
                    'source does not sort before target: {i} > {j}'
                )

    @classmethod
    def get_distance(cls, a, b, method=''):
        """
        Return distance between two objects

        Object names can be provided instead of instances
        """
        model = cls._meta.get_field('source').related_model

        if isinstance(a, str):
            a = model.objects.get(accession=a)

        if isinstance(b, str):
            b = model.objects.get(accession=b)

        source, target = (a, b) if a.accession < b.accession else (b, a)
        obj = cls.objects.get(source=source, target=target, method=method)
        return obj.value


class DistanceMixin():
    """
    Mixin for models that use a m2m field to Distance called neighborhood
    """
    def add_dist(self, other, value, method=Distance.DEFAULT_METHOD):
        attrs = dict(method=method, value=value)
        if self.accession < other.accession:
            self.neighborhood.add(other, through_defaults=attrs)
        else:
            other.neighborhood.add(self, through_defaults=attrs)


class ContigDistance(Distance):
    """
    A (partial) contigs distance matrix

    src.name sorts strictly before dst.name
    """
    source = models.ForeignKey(Contig, **fk_req, related_name='neighbor_to')
    target = models.ForeignKey(Contig, **fk_req, related_name='neighbor_from')


class GeneDistance(Distance):
    """
    A (partial) gene distance matrix

    src.name sorts strictly before dst.name
    """
    source = models.ForeignKey(Gene, **fk_req, related_name='neighbor_to')
    target = models.ForeignKey(Gene, **fk_req, related_name='neighbor_from')


class Gene(Model, DistanceMixin):
    """
    Example how to use DistanceMixin
    """
    STRAND_PLUS = True
    STRAND_MINUS = False
    STRAND_CHOICES = ((STRAND_PLUS, '+'), (STRAND_MINUS, '-'))
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='gene id',
    )
    name = models.CharField(max_length=255)
    contig = models.ForeignKey(Contig, **fk_req)
    start = models.IntegerField()
    end = models.IntegerField()
    strand = models.BooleanField(choices=STRAND_CHOICES)
    neighborhood = models.ManyToManyField(
        'self',
        through='GeneDistance',
        through_fields=('source', 'target'),
        symmetrical=False,
    )



