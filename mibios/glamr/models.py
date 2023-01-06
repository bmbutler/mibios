"""
GLAMR-specific modeling
"""
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models, router, transaction
from django.urls import reverse

from mibios.omics.models import AbstractDataset, AbstractSample
from mibios.umrad.fields import AccessionField
from mibios.umrad.models import Model
from mibios.umrad.model_utils import ch_opt, fk_opt, fk_req, uniq_opt, opt

from .fields import OptionalURLField
from .load import \
    DatasetLoader, ReferenceLoader, SampleLoader, SearchTermManager


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
    bioproject = models.CharField(max_length=100, **ch_opt)
    jgi_project = models.CharField(max_length=100, **ch_opt)
    gold_id = models.CharField(max_length=100, **ch_opt)
    scheme = models.CharField(
        max_length=512,
        **ch_opt,
        verbose_name='location and sampling scheme',
    )
    material_type = models.CharField(
        max_length=128,
        **ch_opt,
    )
    water_bodies = models.CharField(
        max_length=256,
        **ch_opt,
        help_text='list or description of sampled bodies of water',
    )
    primers = models.CharField(
        max_length=64,
        **ch_opt,
    )
    sequencing_target = models.CharField(
        max_length=64,
        **ch_opt,
    )
    sequencing_platform = models.CharField(
        max_length=64,
        **ch_opt,
    )
    size_fraction = models.CharField(
        max_length=32,
        **ch_opt,
        help_text='e.g.: >0.22µm or 0.22-1.6µm',
    )
    note = models.TextField(**ch_opt)

    accession_fields = ('dataset_id', )
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
    short_reference = models.CharField(
        # this field is required
        max_length=128,
        help_text='short reference',
    )
    authors = models.CharField(
        max_length=2048, **ch_opt,
        help_text='author listing',
    )
    title = models.CharField(
        max_length=512, **ch_opt,
    )
    abstract = models.TextField(**ch_opt)
    key_words = models.CharField(max_length=128, **ch_opt)
    publication = models.CharField(max_length=128, **ch_opt)
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

    project_id = models.CharField(
        max_length=32, **ch_opt,
        help_text='Project accession, e.g. NCBI bioproject',
    )
    biosample = models.CharField(max_length=16, **ch_opt)
    geo_loc_name = models.CharField(max_length=256, **ch_opt)
    gaz_id = models.CharField(max_length=16, **ch_opt, verbose_name='GAZ id')
    latitude = models.CharField(max_length=16, **ch_opt)
    longitude = models.CharField(max_length=16, **ch_opt)
    # timestamp: expect ISO8601 formats plus yyyy and yyyy-mm
    collection_timestamp = models.DateTimeField(**opt)
    # Indicate missing time or partial non-ISO6801 dates: e.g. 2013 or 2013-08
    collection_ts_partial = models.CharField(
        max_length=10,
        choices=PARTIAL_TS_CHOICES,
        default=FULL_TIMESTAMP,
        blank=True,
    )
    noaa_site = models.CharField(max_length=16, **ch_opt, verbose_name='NOAA Site')  # noqa: E501
    env_broad_scale = models.CharField(max_length=64, **ch_opt)
    env_local_scale = models.CharField(max_length=64, **ch_opt)
    env_medium = models.CharField(max_length=64, **ch_opt)
    modified_or_experimental = models.BooleanField(default=False)
    depth = models.CharField(max_length=16, **ch_opt)
    depth_sediment = models.CharField(max_length=16, **ch_opt)
    size_frac_up = models.CharField(max_length=16, **ch_opt)
    size_frac_low = models.CharField(max_length=16, **ch_opt)
    ph = models.CharField(max_length=8, **ch_opt, verbose_name='pH')
    temp = models.CharField(max_length=8, **ch_opt)
    calcium = models.CharField(max_length=8, **ch_opt)
    potassium = models.CharField(max_length=8, **ch_opt)
    magnesium = models.CharField(max_length=8, **ch_opt)
    ammonium = models.CharField(max_length=8, **ch_opt)
    nitrate = models.CharField(max_length=8, **ch_opt)
    phosphorus = models.CharField(max_length=8, **ch_opt)
    diss_oxygen = models.CharField(max_length=8, **ch_opt)
    conduc = models.CharField(max_length=16, **ch_opt)
    secci = models.CharField(max_length=8, **ch_opt)
    turbidity = models.CharField(max_length=8, **ch_opt)
    part_microcyst = models.CharField(max_length=8, **ch_opt)
    diss_microcyst = models.CharField(max_length=8, **ch_opt)
    ext_phyco = models.CharField(max_length=8, **ch_opt)
    ext_microcyst = models.CharField(max_length=8, **ch_opt)
    ext_anatox = models.CharField(max_length=8, **ch_opt)
    chlorophyl = models.CharField(max_length=8, **ch_opt)
    diss_phosp = models.CharField(max_length=8, **ch_opt)
    soluble_react_phosp = models.CharField(max_length=8, **ch_opt)
    ammonia = models.CharField(max_length=8, **ch_opt)
    nitrate_nitrite = models.CharField(max_length=8, **ch_opt)
    urea = models.CharField(max_length=8, **ch_opt)
    part_org_carb = models.CharField(max_length=8, **ch_opt)
    part_org_nitro = models.CharField(max_length=8, **ch_opt)
    diss_org_carb = models.CharField(max_length=8, **ch_opt)
    col_dom = models.CharField(max_length=8, **ch_opt)
    suspend_part_matter = models.CharField(max_length=8, **ch_opt)
    suspend_vol_solid = models.CharField(max_length=8, **ch_opt)
    microcystis_count = models.PositiveIntegerField(**opt)
    planktothris_count = models.PositiveIntegerField(**opt)
    anabaena_d_count = models.PositiveIntegerField(**opt)
    cylindrospermopsis_count = models.PositiveIntegerField(**opt)
    notes = models.CharField(max_length=512, **ch_opt)

    loader = SampleLoader()

    class Meta:
        default_manager_name = 'objects'

    def __str__(self):
        return self.sample_name or self.sample_id or self.biosample \
            or super().__str__()


class SearchTerm(models.Model):
    term = models.CharField(max_length=32, db_index=True)
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
