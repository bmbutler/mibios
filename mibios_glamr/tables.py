from django_tables2 import Table, Column


class DatasetTable(Table):
    samples = Column(
        verbose_name='available samples',
        linkify=lambda record: record.get_samples_url,
    )
    scheme = Column(
        verbose_name='description',
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
        template_name = 'django_tables2/bootstrap.html'

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
