"""
GLAMR-specific modeling
"""
from django.db import models

from mibios.models import Model
from mibios_omics.models import AbstractSampleGroup
from mibios_umrad.model_utils import fk_req

from .load import DatasetLoader


class Dataset(AbstractSampleGroup):
    """
    A collection of related samples, e.g. a study or project
    """
    # FIXME: it's not clear which field identifies a "data set", which field
    # may not be blank, and which shall be unique
    reference = models.ForeignKey('Reference', **fk_req)
    DB_GLAMR = 'GLAMR',
    DB_NCBI = 'NCBI'
    DB_JGI = 'JGI'
    DB_CAMERA = 'Camera'
    DB_VAMPS = 'VAMPS'
    ACCESSION_DB_CHOICES = (
        ('NCBI', DB_NCBI),
        ('JGI', DB_JGI),
        ('Camera', DB_CAMERA),
        ('VAMPS', DB_VAMPS),
    )
    accession = models.CharField(
        max_length=32,
        blank=True,
        help_text='accession to data set/study/project',
    )
    accession_db = models.CharField(
        max_length=8,
        blank=True,
        choices=ACCESSION_DB_CHOICES,
        help_text='Database associated with accession',
    )
    scheme = models.CharField(
        max_length=512,
        blank=False,
        verbose_name='location and sampling scheme',
    )
    material_type = models.CharField(
        max_length=128,
    )
    water_bodies = models.CharField(
        max_length=256,
        help_text='list or description of sampled bodies of water',
    )
    primers = models.CharField(
        max_length=64,
        blank=True,
    )
    gene_target = models.CharField(
        max_length=64,
        blank=True,
    )
    sequencing_platform = models.CharField(
        max_length=64,
        blank=True,
    )
    size_fraction = models.CharField(
        max_length=32,
        blank=True,
        help_text='e.g.: >0.22µm or 0.22-1.6µm',
    )
    note = models.TextField(blank=True)

    loader = DatasetLoader()
    orphan_group_description = 'samples without a data set'

    class Meta:
        default_manager_name = 'objects'
        unique_together = (
            'reference', 'accession', 'accession_db',
        )

    def __init__(self, *args, orphan_group=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.orphan_group = orphan_group
        if not self.scheme:
            self.scheme = self.orphan_group_description

    def __str__(self):
        if self.orphan_group:
            return self.orphan_group_description
        if self.reference_id is None:
            ref_str = ''
        else:
            ref_str = self.reference.short_reference
        maxlen = 60 - len(ref_str)  # max length available for scheme part
        scheme_str = self.scheme
        if len(scheme_str) > maxlen:
            scheme_str = scheme_str[:maxlen]
            # remove last word and add [...]
            scheme_str = ' '.join(scheme_str.split(' ')[:-1]) + '[\u2026]'

        return ' - '.join(filter(None, [scheme_str, ref_str]))

    def get_accession_url(self):
        if self.accession_db == self.DB_NCBI:
            return f'https://www.ncbi.nlm.nih.gov/search/all/?term={self.accession}'  # noqa: E501
        elif self.accession_db == self.DB_JGI:
            return f'https://genome.jgi.doe.gov/portal/?core=genome&query={self.accession}'  # noqa: E501


# Create your models here.
class Reference(Model):
    """
    A journal article or similar, primary reference for a data set
    """
    short_reference = models.CharField(
        max_length=128,
        blank=False,
        help_text='short reference',
    )
    authors = models.CharField(
        max_length=2048,
        blank=True,
        help_text='author listing',
    )
    title = models.CharField(
        max_length=512,
        blank=True,
    )
    abstract = models.TextField(blank=True)
    key_words = models.CharField(max_length=128, blank=True)
    publication = models.CharField(max_length=128)
    doi = models.URLField()

    class Meta:
        unique_together = (
            # FIXME: this, for now, needs fix in source data
            ('short_reference', 'publication'),
        )

    def __str__(self):
        maxlen = 60
        value = f'{self.short_reference} "{self.title}"'
        if len(value) > maxlen:
            value = value[:maxlen]
            # rm last word, add ellipsis
            value = ' '.join(value.split(' ')[:-1]) + '[\u2026]"'
        return value
