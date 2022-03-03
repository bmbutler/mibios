from django.urls import reverse

from django_tables2 import Table, Column


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

    class Meta:
        template_name = 'django_tables2/bootstrap4.html'
        pass

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
        return f'{record.samples().count()}'


def get_sample_url(sample):
    """ linkify helper for SampleTable """
    return reverse('sample_detail', args=[sample.pk])


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
