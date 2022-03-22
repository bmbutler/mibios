from django.urls import reverse

from django_tables2 import Column, Table, TemplateColumn

from mibios_glamr import models as glamr_models
from mibios_omics import models as omics_models


def get_record_url(*args):
    """
    Return URL for an object

    Arguments: <obj> | <<model|model_name> <pk>>

    The object can be passed as the only argument.  Or the model/model name and
    PK must be passed.

    Use this instead of Model.get_absolute_url() because it needs to work on
    models from other apps.
    """
    if len(args) == 1:
        obj = args[0]
        model_name = obj._meta.model_name
        pk = obj.pk
    elif len(args) == 2:
        model, pk = args
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model._meta.model_name
    else:
        raise TypeError(
            'expect either a model instance or model/-name and pk pair'
        )
    return reverse('record', kwargs={'model': model_name, 'pk': pk})


class CompoundAbundanceTable(Table):
    sample = Column(
        linkify=lambda value: get_record_url(value)
    )
    compound = Column(
        linkify=lambda value: get_record_url(value)
    )

    class Meta:
        model = omics_models.CompoundAbundance
        exclude = ['id']


class FunctionAbundanceTable(Table):
    class Meta:
        model = omics_models.FuncAbundance
        exclude = ['id']


class OverViewTable(Table):
    num_samples = TemplateColumn(
        """<a href="{% url 'record_overview_samples' model=table.view_object_model_name pk=table.view_object.pk %}">{{ value }}</a> out of {{ record.total_samples }}""",  # noqa: E501
        verbose_name='number of samples',
    )
    short = TemplateColumn(
        "{{ record }}",
        linkify=lambda record: get_record_url(record),
        verbose_name='mini description',
    )

    class Meta:
        model = glamr_models.Dataset
        fields = [
            'num_samples', 'short', 'water_bodies', 'year', 'Institution/PI',
            'sequencing_data_type',
        ]


class OverViewSamplesTable(Table):
    accession = Column(
        linkify=lambda record: get_record_url(record),
        verbose_name='sample',
    )
    sample_name = Column(verbose_name='other names')
    group = Column(
        linkify=lambda value: get_record_url(value),
        verbose_name='dataset',
    )

    class Meta:
        model = glamr_models.Sample
        fields = [
            'accession', 'sample_name', 'group', 'group.water_bodies',
            'date', 'Institution/PI', 'latitude', 'longitude',
        ]


class TaxonAbundanceTable(Table):
    sample = Column(
        linkify=lambda value: get_record_url('sample', value.pk)
    )

    class Meta:
        model = omics_models.TaxonAbundance
        fields = ['sample', 'lin_avg_rpkm', 'lin_gnm_pgc', 'lin_sum_sco']


class DatasetTable(Table):
    samples = Column(
        verbose_name='available samples',
        linkify=lambda record: record.get_samples_url,
    )
    scheme = Column(
        verbose_name='description',
        linkify=True,
    )
    reference = Column(
        linkify=lambda value: getattr(value, 'doi'),
    )
    accession = Column(
        verbose_name='data repository',
        linkify=lambda record: record.get_accession_url(),
    )
    gene_target = Column(
        verbose_name='other info',
    )

    def render_scheme(self, value, record):
        if record.material_type:
            value += f' | {record.material_type}'
        if record.water_bodies:
            value += f' @ {record.water_bodies}'
        return value

    def render_accession(self, record):
        return f'{record.accession_db}: {record.accession}'

    def render_other(self, record):
        return f'{record.gene_target} {record.size_fraction}'

    def render_samples(self, record):
        return f'{record.sample_count}'


def get_sample_url(sample):
    """ linkify helper for SampleTable """
    return reverse('sample_detail', args=[sample.pk])


class SingleColumnRelatedTable(Table):
    """ Table showing *-to-many related records in single column """
    objects = Column(
        verbose_name='related records',
        linkify=lambda record: get_record_url(record),
        empty_values=(),  # triggers render_objects()
    )

    def render_objects(self, record):
        return str(record)


class SampleTable(Table):

    accession = Column(
        verbose_name='accession',  # FIXME: verbose_name
        # linkify=lambda record: get_sample_url(record),
        linkify=lambda record: reverse('sample', args=[record.pk]),
    )
    group = Column(verbose_name='dataset')
    read_count = Column()
    reads_mapped_contigs = Column()
    reads_mapped_genes = Column()
