"""
GLAMR-specific modeling
"""
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models, router, transaction
from django.urls import reverse

from mibios.omics.models import AbstractDataset, AbstractSample
from mibios.omics.managers import SampleManager
from mibios.umrad.fields import AccessionField
from mibios.umrad.manager import Manager
from mibios.umrad.models import Model
from mibios.umrad.model_utils import ch_opt, fk_opt, fk_req, uniq_opt, opt

from .fields import OptionalURLField
from .load import \
    DatasetLoader, ReferenceLoader, SampleLoader, SearchTermManager
from .queryset import DatasetQuerySet, SampleQuerySet


class Dataset(AbstractDataset):
    """
    A collection of related samples, e.g. a study or project
    """
    dataset_id = AccessionField(
        # overrides abstract parent field
        unique=True,
        verbose_name='Dataset ID',
        help_text='GLAMR accession to data set/study/project',
    )
    reference = models.ForeignKey('Reference', **fk_opt)
    # project IDs: usually a single accession, but can be ,-sep lists or even
    # other text
    bioproject = models.TextField(max_length=32, **ch_opt)
    jgi_project = models.TextField(max_length=32, **ch_opt)
    gold_id = models.TextField(max_length=32, **ch_opt)
    scheme = models.TextField(
        **ch_opt,
        verbose_name='location and sampling scheme',
    )
    material_type = models.TextField(
        **ch_opt,
    )
    water_bodies = models.TextField(
        **ch_opt,
        help_text='list or description of sampled bodies of water',
    )
    primers = models.TextField(
        max_length=32,
        **ch_opt,
    )
    sequencing_target = models.TextField(
        max_length=32,
        **ch_opt,
    )
    sequencing_platform = models.TextField(
        max_length=32,
        **ch_opt,
    )
    size_fraction = models.TextField(
        max_length=32,
        **ch_opt,
        help_text='e.g.: >0.22µm or 0.22-1.6µm',
    )
    note = models.TextField(**ch_opt)

    accession_fields = ('dataset_id', )

    objects = Manager.from_queryset(DatasetQuerySet)()
    loader = DatasetLoader()
    orphan_group_description = 'Samples without a data set'

    class Meta:
        default_manager_name = 'objects'

    def __init__(self, *args, orphan_group=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.orphan_group = orphan_group
        if orphan_group and not self.short_name:
            self.short_name = self.orphan_group_description

    def __str__(self):
        if self.reference_id is None:
            ref = ''
        else:
            ref = self.reference.short_reference
        maxlen = 60 - len(ref)  # max length available for scheme part
        scheme = self.scheme
        if scheme and len(scheme) > maxlen:
            scheme = scheme[:maxlen]
            # remove last word and add [...]
            scheme = ' '.join(scheme.split(' ')[:-1]) + '[\u2026]'

        return ' - '.join(filter(None, [scheme, ref])) or self.short_name \
            or super().__str__()

    bioproject_url_templ = 'https://www.ncbi.nlm.nih.gov/bioproject/{}'
    jgi_project_url_templ = 'https://genome.jgi.doe.gov/portal/lookup' \
        '?keyName=jgiProjectId&keyValue={}&app=Info&showParent=false'
    gold_id_url_templ = None

    def external_urls(self):
        """ collect all external accessions with URLs """
        urls = []
        for i in ['bioproject', 'jgi_project', 'gold_id']:
            field_value = getattr(self, i)
            if field_value is None:
                items = []
            else:
                items = field_value.replace(',', ' ').split()
            for j in items:
                template = getattr(self, i + '_url_templ')
                if i == 'bioproject' and not j.startswith('PRJ'):
                    # TODO: bad accession, fix at source
                    template = None
                if template:
                    urls.append((j, template.format(j)))
                else:
                    urls.append((j, None))
        return urls

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


class Reference(Model):
    """
    A journal article or similar, primary reference for a data set
    """
    reference_id = AccessionField(prefix='paper_')
    short_reference = models.TextField(
        # this field is required
        max_length=32,
        help_text='short reference',
    )
    authors = models.TextField(
        **ch_opt,
        help_text='author listing',
    )
    title = models.TextField(**ch_opt)
    abstract = models.TextField(**ch_opt)
    key_words = models.TextField(**ch_opt)
    publication = models.TextField(max_length=64, **ch_opt)
    doi = OptionalURLField(**uniq_opt)

    loader = ReferenceLoader()

    class Meta:
        pass

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
    DATE_ONLY = 'date_only'
    YEAR_ONLY = 'year_only'
    MONTH_ONLY = 'month_only'
    FULL_TIMESTAMP = ''
    PARTIAL_TS_CHOICES = (
        (DATE_ONLY, DATE_ONLY),
        (YEAR_ONLY, YEAR_ONLY),
        (MONTH_ONLY, MONTH_ONLY),
        (FULL_TIMESTAMP, FULL_TIMESTAMP),
    )

    project_id = models.TextField(
        max_length=32, **ch_opt,
        help_text='Project accession, e.g. NCBI bioproject',
    )
    biosample = models.TextField(max_length=32, **ch_opt)
    geo_loc_name = models.TextField(max_length=64, **ch_opt)
    gaz_id = models.TextField(max_length=32, **ch_opt, verbose_name='GAZ id')
    latitude = models.TextField(max_length=16, **ch_opt)
    longitude = models.TextField(max_length=16, **ch_opt)
    # timestamp: expect ISO8601 formats plus yyyy and yyyy-mm
    collection_timestamp = models.DateTimeField(**opt)
    # Indicate missing time or partial non-ISO6801 dates: e.g. 2013 or 2013-08
    collection_ts_partial = models.CharField(
        max_length=10,
        choices=PARTIAL_TS_CHOICES,
        default=FULL_TIMESTAMP,
        blank=True,
    )
    noaa_site = models.TextField(max_length=16, **ch_opt, verbose_name='NOAA Site')  # noqa: E501
    env_broad_scale = models.TextField(max_length=32, **ch_opt)
    env_local_scale = models.TextField(max_length=32, **ch_opt)
    env_medium = models.TextField(max_length=32, **ch_opt)
    keywords = models.TextField(max_length=32, **ch_opt)
    depth = models.TextField(max_length=16, **ch_opt)
    depth_sediment = models.TextField(max_length=16, **ch_opt)
    size_frac_up = models.TextField(max_length=16, **ch_opt)
    size_frac_low = models.TextField(max_length=16, **ch_opt)
    ph = models.TextField(max_length=8, **ch_opt, verbose_name='pH')
    temp = models.TextField(max_length=8, **ch_opt)
    calcium = models.TextField(max_length=8, **ch_opt)
    potassium = models.TextField(max_length=8, **ch_opt)
    magnesium = models.TextField(max_length=8, **ch_opt)
    ammonium = models.TextField(max_length=8, **ch_opt)
    nitrate = models.TextField(max_length=8, **ch_opt)
    total_phos = models.TextField(max_length=8, **ch_opt)
    diss_oxygen = models.TextField(max_length=8, **ch_opt)
    conduc = models.TextField(max_length=16, **ch_opt)
    secci = models.TextField(max_length=8, **ch_opt)
    turbidity = models.TextField(max_length=8, **ch_opt)
    part_microcyst = models.TextField(max_length=8, **ch_opt)
    diss_microcyst = models.TextField(max_length=8, **ch_opt)
    ext_phyco = models.TextField(max_length=8, **ch_opt)
    ext_microcyst = models.TextField(max_length=8, **ch_opt)
    ext_anatox = models.TextField(max_length=8, **ch_opt)
    chlorophyl = models.TextField(max_length=8, **ch_opt)
    diss_phos = models.TextField(max_length=8, **ch_opt)
    soluble_react_phos = models.TextField(max_length=8, **ch_opt)
    ammonia = models.TextField(max_length=8, **ch_opt)
    nitrate_nitrite = models.TextField(max_length=8, **ch_opt)
    urea = models.TextField(max_length=8, **ch_opt)
    part_org_carb = models.TextField(max_length=8, **ch_opt)
    part_org_nitro = models.TextField(max_length=8, **ch_opt)
    diss_org_carb = models.TextField(max_length=8, **ch_opt)
    col_dom = models.TextField(max_length=8, **ch_opt)
    suspend_part_matter = models.TextField(max_length=8, **ch_opt)
    suspend_vol_solid = models.TextField(max_length=8, **ch_opt)
    microcystis_count = models.PositiveIntegerField(**opt)
    planktothrix_count = models.PositiveIntegerField(**opt)
    anabaena_d_count = models.PositiveIntegerField(**opt)
    cylindrospermopsis_count = models.PositiveIntegerField(**opt)
    ice_cover = models.PositiveSmallIntegerField(**opt)
    chlorophyl_fluoresence = models.PositiveSmallIntegerField(**opt)
    sampling_device = models.TextField(max_length=32, **ch_opt)
    modified_or_experimental = models.BooleanField(default=False)
    is_isolate = models.BooleanField(**opt)
    is_neg_control = models.BooleanField(**opt)
    is_pos_control = models.BooleanField(**opt)
    filt_volume = models.DecimalField(max_digits=10, decimal_places=3, **opt)
    filt_duration = models.DurationField(**opt)
    par = models.DecimalField(max_digits=8, decimal_places=2, **opt)
    qPCR_total = models.PositiveIntegerField(**opt)
    qPCR_mcyE = models.PositiveIntegerField(**opt)
    qPCR_sxtA = models.PositiveIntegerField(**opt)
    notes = models.TextField(**ch_opt)

    objects = SampleManager.from_queryset(SampleQuerySet)()
    loader = SampleLoader()

    class Meta:
        default_manager_name = 'objects'

    def __str__(self):
        return self.sample_name or self.sample_id or self.biosample \
            or super().__str__()


class SearchTerm(models.Model):
    term = models.TextField(max_length=32, db_index=True)
    has_hit = models.BooleanField(default=False)
    content_type = models.ForeignKey(ContentType, **fk_req)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')

    objects = SearchTermManager()

    def __str__(self):
        return self.term


def load_meta_data(dry_run=False):
    """
    load meta data assuming empty DB -- reference implementation
    """
    dbalias = router.db_for_write(Sample)
    with transaction.atomic(using=dbalias):
        Reference.loader.load()
        Dataset.loader.load()
        Sample.loader.load_meta()
        # Sample.objects.sync()
        if dry_run:
            transaction.set_rollback(True, dbalias)
