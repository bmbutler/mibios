"""
GLAMR-specific modeling
"""
from django.db import models
from django.urls import reverse

from mibios_omics.models import AbstractSampleGroup, AbstractSample
from mibios_umrad.models import Model
from mibios_umrad.model_utils import fk_opt

from .load import DatasetLoader, SampleLoader


class Dataset(AbstractSampleGroup):
    """
    A collection of related samples, e.g. a study or project
    """
    # FIXME: it's not clear which field identifies a "data set", which field
    # may not be blank, and which shall be unique
    reference = models.ForeignKey('Reference', **fk_opt)
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
        blank=True,
        verbose_name='location and sampling scheme',
    )
    sequencing_data_type = models.CharField(
        max_length=128,
        blank=True,
        help_text='e.g. amplicon, metagenome',
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
        if not self.short_name:
            self.short_name = self.orphan_group_description

    def __str__(self):
        if self.orphan_group:
            return self.orphan_group_description
        if self.reference_id is None:
            ref = ''
        else:
            ref = self.reference.short_reference
        maxlen = 60 - len(ref)  # max length available for scheme part
        scheme = self.scheme
        if len(scheme) > maxlen:
            scheme = scheme[:maxlen]
            # remove last word and add [...]
            scheme = ' '.join(scheme.split(' ')[:-1]) + '[\u2026]'

        return ' - '.join(filter(None, [scheme, ref])) or self.short_name

    def get_accession_url(self):
        if self.accession_db == self.DB_NCBI:
            return f'https://www.ncbi.nlm.nih.gov/search/all/?term={self.accession}'  # noqa: E501
        elif self.accession_db == self.DB_JGI:
            return f'https://genome.jgi.doe.gov/portal/?core=genome&query={self.accession}'  # noqa: E501

    def get_absolute_url(self):
        if self.orphan_group:
            return reverse('dataset', args=[0])
        return reverse('dataset', args=[self.pk])

    def get_samples_url(self):
        pk = 0 if self.orphan_group else self.pk
        return reverse('dataset_sample_list', args=[pk])


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

    def get_absolute_url(self):
        return reverse('reference', args=[self.pk])


class Sample(AbstractSample):
    site = models.CharField(max_length=64, blank=True, verbose_name='Site')
    fraction = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='fraction',
    )
    sample_name = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Sample_name',
    )
    date = models.DateField(verbose_name='Date', blank=True, null=True)
    station_depth = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Station Depth (m)',
    )
    sample_depth = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Sample Depth (m)',
    )
    sample_depth_category = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Sample Depth (category)',
    )
    local_time = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Local Time (Eastern Time Zone)',
    )
    latitude = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Latitude (decimal deg)',
    )
    longitude = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Longitude (decimal deg)',
    )
    wind_speed = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Wind speed (knots)',
    )
    wave_height = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Wave Height (ft)',
    )
    sky = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Sky',
    )
    secchi_depth = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Secchi Depth (m)',
    )
    sample_temperature = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Sample Temperature (°C)',
    )
    ctd_temperature = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='CTD Temperature (°C)',
    )
    ctd_specific_conductivity = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='CTD Specific Conductivity (µS/cm)',
    )
    ctd_beam_attenuation = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='CTD Beam Attenuation (m-1)',
    )
    ctd_tramission = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='CTD Tramission (%)',
    )
    ctd_dissolved_oxygen = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='CTD Dissolved Oxygen (mg/L)',
    )
    ctd_radiation = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='CTD Photosynthetically Active Radiation (µE/m2/s)',
    )
    turbidity = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Turbidity (NTU)',
    )
    particulate_microcystin = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Particulate Microcystin (µg/L)',
    )
    dissolved_microcystin = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Dissolved Microcystin (µg/L)',
    )
    extracted_phycocyanin = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Extracted Phycocyanin (µg/L)',
    )
    extracted_chlorophyll_a = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Extracted Chlorophyll a (µg/L)',
    )
    phosphorus = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Total Phosphorus (µg P/L)',
    )
    dissolved_phosphorus = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Total Dissolved Phosphorus (µg P/L)',
    )
    soluble_reactive_phosphorus = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Soluble Reactive Phosphorus (µg P/L)',
    )
    ammonia = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Ammonia (µg N/L)',
    )
    nitrate_nitrite = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Nitrate + Nitrite (mg N/L)',
    )
    urea = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Urea (µg N/L)',
    )
    organic_carbon = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Particulate Organic Carbon (mg/L)',
    )
    organic_nitrogen = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Particulate Organic Nitrogen (mg/L)',
    )
    dissolved_organic_carbon = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Dissolved Organic Carbon (mg/L)',
    )
    absorbance = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Colored Dissolved Organic Material absorbance (m-1) at '
                     '400nm',
    )
    suspended_solids = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Total Suspended Solids (mg/L)',
    )
    Volatile_suspended_solids = models.CharField(
        max_length=64,
        blank=True,
        verbose_name='Volatile Suspended Solids (mg/L)',
    )

    loader = SampleLoader()
