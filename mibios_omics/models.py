from django.core.exceptions import ValidationError
from django.db import models

from mibios.models import Model, ParentModel


# standard data field options
opt = dict(blank=True, null=True, default=None)  # non-char optional
ch_opt = dict(blank=True, default='')  # optional char
uniq_opt = dict(unique=True, **opt)  # unique and optional (char/non-char)
# standard foreign key options
fk_req = dict(on_delete=models.CASCADE)  # required FK
fk_opt = dict(on_delete=models.SET_NULL, **opt)  # optional FK

# functional DB sources
FUNC_DB_KEGG = 'KEGG'
FUNC_DB_CHEBI = 'CHEBI'
FUNC_DB_RHEA = 'RHEA'
FUNC_DB_BIOCYC = 'biocyc'


class NullCharField(models.CharField):
    """
    A nullable char field

    If used with null=True, then empty strings and other empty values will be
    stored as NULL, and NULLs at the db level will be returned as empty string.
    This allows to meaningfully combine unique=True and blank=True which
    otherwise, with the regular CharField, is not possible, having multiple
    blanks would violate the uniquness contraint.

    If null=False this class behaves exactly like CharField.
    """
    def to_python(self, value):
        if self.null and value in self.empty_values:
            return ''
        return super().to_python(value)

    def get_db_prep_value(self, value, *args, **kwargs):
        if self.null and value in self.empty_values:
            return None
        else:
            return super().get_db_prep_value(value, *args, **kwargs)

    def from_db_value(self, value, expression, connection):
        if value is None:
            return ''
        else:
            return value  # assume a str


class SecondarySample(ParentModel):
    """
    A specimen that has undergone some type of further analysis

    E.g. sequencing, metabolome, metaproteome analysis

    It is derived from a primary sample.
    """
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='sample id',
    )
    public_accession = models.CharField(max_length=32, **ch_opt)

    # primary sample and how it's derived from it:
    sample = models.ForeignKey('Sample', **fk_req)
    size_fraction = models.CharField(max_length=32, **ch_opt)

    # how sequencing was done:
    seq_type = models.CharField(
        max_length=32, verbose_name='sequencing project type',
    )


class SequencedSample(SecondarySample):
    """
    A sample that underwent sequencing
    """
    seq_project = models.ForeignKey('SequencingProject', **fk_req)
    reads_location = models.CharField(
        max_length=255, **ch_opt,
        help_text='URL, path, SRA accession, or similar to one or multiple '
                  'FASTQ files. Can be glob or regex.',
    )
    read_yield = models.PositiveIntegerField(
        verbose_name='total read count / yield',
    )


class AmpliconSample(SequencedSample):
    pass


class Analysis(Model):
    """
    A data analysis project

    E.g. assembly, OTU synthesis.  This is what connects a set of
    samples/fastq-reads to contigs/OTUs/abundance.
    """
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='analysis id',
    )
    method = models.CharField(
        max_length=64,
        help_text='name, short description, or reference to SOP',
    )
    description = models.TextField(
        blank=True,
        help_text='free-form description of SOP',
    )
    sample = models.ManyToManyField('SequencedSample')

    class Meta:
        verbose_name_plural = 'analyses'


class CollectionSite(Model):
    """
    Sample collection site

    site of environmental sample collection or location of host
    """
    name = models.CharField(
        max_length=255, unique=True, **opt,
        verbose_name='site name',
        help_text='unique site identifier',
    )
    location = models.CharField(
        max_length=255, **ch_opt,
        verbose_name='geographic location',
        help_text='Geographic name, name of lake or city',
    )
    location_description = models.CharField(
        max_length=255, **ch_opt,
        help_text='Further description of location',
    )
    latitude = models.DecimalField(
        max_digits=15, decimal_places=12, **opt,
        help_text='latitude in decimal degrees'
    )
    longitude = models.DecimalField(
        max_digits=15, decimal_places=12, **opt,
        help_text='longitude in decimal degrees'
    )
    depth = models.PositiveSmallIntegerField(
        **opt, help_text='water depth in meters',
    )

    class Meta:
        unique_together = (
            ('latitude', 'longitude'),
        )


class Compound(Model):
    COMPOUND_SOURCE = (
        (FUNC_DB_KEGG, FUNC_DB_KEGG),
        (FUNC_DB_CHEBI, FUNC_DB_CHEBI),
        (FUNC_DB_BIOCYC, FUNC_DB_BIOCYC),
    )
    source_accession = models.CharField(max_length=32)
    source_db = models.CharField(
        max_length=32,
        choices=COMPOUND_SOURCE,
    )
    name = models.CharField(
        max_length=255,
        help_text='human readable name',
    )
    mol_mass = models.DecimalField(
        max_digits=15, decimal_places=12, **opt,
    )
    formula = models.CharField(max_length=64, **ch_opt)
    charge = models.CharField(max_length=16, **ch_opt)

    class Meta:
        unique_together = (
            ('source_accession', 'source_db'),
        )


class CompoundAbundance(Model):
    history = None
    compound = models.ForeignKey(Compound, **fk_req)
    sample = models.ForeignKey('MetabolomeSample', **fk_req)
    m_over_z = models.FloatField(verbose_name='m/z')
    intensity = models.CharField(max_length=32)
    retention_time = models.DecimalField(
        max_digits=5, decimal_places=3,
    )
    relative = models.DecimalField(
        max_digits=5, decimal_places=3,
        verbose_name='relative abundance',
    )


class Contig(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='contig id',
    )
    sequence = models.ForeignKey('Sequence', **fk_req)
    analysis = models.ForeignKey(Analysis, **fk_req)


class ContigAbundance(Model):
    history = None
    contig = models.ForeignKey(Contig, **fk_req)
    sample = models.ForeignKey('SequencedSample', **fk_req)
    absolute = models.FloatField(verbose_name='avg fold')
    relative = models.FloatField(verbose_name='fpkm')
    cover = models.FloatField(verbose_name='% covered')
    analysis = models.ForeignKey('Analysis', **fk_req)


class ContigCluster(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='contig cluster id',
    )
    level = models.ForeignKey('ContigClusterLevel', **fk_req)
    rep = models.ForeignKey(Contig, **fk_req, related_name='representing')
    members = models.ManyToManyField(
        Contig,
        through='ContigClusterMembership',
    )


class ContigClusterLevel(Model):
    label = models.CharField(max_length=16, unique=True)
    ani = models.SmallIntegerField(verbose_name='% ANI')
    cov = models.SmallIntegerField(verbose_name='% coverage')


class ContigClusterMembership(Model):
    cluster = models.ForeignKey(ContigCluster, **fk_req)
    contig = models.ForeignKey(Contig, **fk_req)
    ani = models.DecimalField(max_digits=6, decimal_places=3)
    cov = models.DecimalField(max_digits=6, decimal_places=3)
    history = None

    class Meta:
        unique_together = (('cluster', 'contig'),)


class Gene(Model):
    STRAND_PLUS = True
    STRAND_MINUS = False
    STRAND_CHOICES = ((STRAND_PLUS, '+'), (STRAND_MINUS, '-'))
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='gene id',
    )
    name = models.CharField(max_length=255)
    gene_type = models.CharField(
        max_length=32,
        # TODO: choices=...
        help_text='choice of asv, xrna, protein, ...',
    )
    contig = models.ForeignKey(Contig, **fk_req)
    start = models.IntegerField()
    end = models.IntegerField()
    strand = models.BooleanField(choices=STRAND_CHOICES)
    length = models.PositiveIntegerField()
    gc_content = models.FloatField(verbose_name='% GC')
    partial_start = models.BooleanField()
    partial_end = models.BooleanField()
    taxon = models.ManyToManyField('Taxonomy')
    lca = models.ForeignKey('Taxonomy', **fk_req, related_name='gene_with_lca')
    uniref100 = models.ForeignKey('UniRef100', **fk_req)  # same as protein ??
    rna_db = models.ForeignKey('RNA_DB', **fk_req)


class GeneAbundance(Model):
    # TODO: how is this different from contig abundance?
    history = None
    gene = models.ForeignKey(Gene, **fk_req)
    sample = models.ForeignKey('SequencedSample', **fk_req)
    absolute = models.FloatField(verbose_name='avg fold')
    relative = models.FloatField(
        help_text='RPKM/FPKM (or % for amplicon?)'
    )
    cover = models.FloatField(verbose_name='% covered')
    analysis = models.ForeignKey('Analysis', **fk_req)


class GeneCluster(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='gene cluster id',
    )
    level = models.ForeignKey('GeneClusterLevel', **fk_req)
    rep = models.ForeignKey(Gene, **fk_req, related_name='representing')
    rna_db = models.ForeignKey(
        'RNA_DB', **fk_opt,
        help_text='RNA DB top hit',
    )
    uniref100 = models.ForeignKey(
        'UniRef100', **fk_req,
        help_text='UniRef100 top hit',
    )  # same as for proteins???
    members = models.ManyToManyField(
        Gene,
        through='GeneClusterMembership',
    )


class GeneClusterLevel(Model):
    label = models.CharField(max_length=16, unique=True)
    ani = models.SmallIntegerField(verbose_name='% ANI')
    cov = models.SmallIntegerField(verbose_name='% coverage')


class GeneClusterMembership(Model):
    cluster = models.ForeignKey(GeneCluster, **fk_req)
    gene = models.ForeignKey(Gene, **fk_req)
    ani = models.DecimalField(max_digits=6, decimal_places=3)
    cov = models.DecimalField(max_digits=6, decimal_places=3)
    history = None

    class Meta:
        unique_together = (('cluster', 'gene'),)


class MetagenomeSample(SequencedSample):
    pass


class MetabolomeSample(SecondarySample):
    ms_type = models.CharField(max_length=32, **ch_opt)
    is_targeted = models.BooleanField(
        verbose_name='is targeted (or whole)',
    )
    spike_in = models.CharField(
        max_length=64,
        verbose_name='spike-in/standard', **ch_opt,
    )


class MetaproteomeSample(SecondarySample):
    ms_type = models.CharField(max_length=32, **ch_opt)
    is_labeled = models.BooleanField(
        verbose_name='labeled',
    )
    spike_in = models.CharField(
        max_length=64,
        verbose_name='spike-in/standard', **ch_opt,
    )


class MetaTranscriptAbundance(Model):
    history = None
    gene_sequence = models.ForeignKey('Sequence', **fk_req)
    sample = models.ForeignKey('MetaTranscriptomeSample', **fk_req)
    absolute = models.FloatField(verbose_name='avg fold')
    relative = models.FloatField(
        help_text='RPKM/FPKM (or % for amplicon?)'
    )
    cover = models.FloatField(verbose_name='% covered')
    analysis = models.ForeignKey('Analysis', **fk_req)


class MetaTranscriptomeSample(SequencedSample):
    pass


class Pathway(Model):
    PATHWAY_SOURCE = (
        (FUNC_DB_KEGG, FUNC_DB_KEGG),
        (FUNC_DB_BIOCYC, FUNC_DB_BIOCYC),
    )
    source_accession = models.CharField(max_length=32)
    source_db = models.CharField(max_length=16, choices=PATHWAY_SOURCE)


class PeptideAbundance(Model):
    history = None
    protein_sequence = models.ForeignKey('ProteinSequence', **fk_req)
    sample = models.ForeignKey('MetaproteomeSample', **fk_req)
    relative = models.DecimalField(
        max_digits=6, decimal_places=3,
        verbose_name='relative intensity'
    )
    cover = models.FloatField(verbose_name='% covered')
    analysis = models.ForeignKey('Analysis', **fk_req)


class Protein(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='protein id',
    )
    gene = models.ForeignKey(Gene, **fk_req)  # or is it 1-to-1 ??
    sequence = models.ForeignKey('ProteinSequence', **fk_req)
    reaction = models.ManyToManyField('ReactionEquivGroup')
    length = models.PositiveIntegerField()
    partial = models.CharField(max_length=8, **ch_opt)
    uniref100 = models.ManyToManyField('UniRef100')
    taxon = models.ManyToManyField('Taxonomy')
    lca = models.ForeignKey(
        'Taxonomy', **fk_req,
        related_name='protein_with_lca',
    )


class ProteinAbundance(Model):
    # TODO: how is this different from gene abundance?
    history = None
    Protein = models.ForeignKey(Protein, **fk_req)
    sample = models.ForeignKey('SequencedSample', **fk_req)
    absolute = models.FloatField(verbose_name='avg fold')
    relative = models.FloatField(
        help_text='RPKM/FPKM'
    )
    cover = models.FloatField(verbose_name='% covered')
    analysis = models.ForeignKey('Analysis', **fk_req)


class ProteinCluster(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='protein cluster id',
    )
    level = models.ForeignKey('ProteinClusterLevel', **fk_req)
    rep = models.ForeignKey(Protein, **fk_req, related_name='representing')
    uniref100 = models.ForeignKey(
        'UniRef100', **fk_req,
        help_text='UniRef100 top hit',
    )
    members = models.ManyToManyField(
        Protein,
        through='ProteinClusterMembership',
    )


class ProteinClusterLevel(Model):
    label = models.CharField(max_length=16, unique=True)
    ani = models.SmallIntegerField(verbose_name='% ANI')
    cov = models.SmallIntegerField(verbose_name='% coverage')


class ProteinClusterMembership(Model):
    cluster = models.ForeignKey(ProteinCluster, **fk_req)
    protein = models.ForeignKey(Protein, **fk_req)
    ani = models.DecimalField(max_digits=6, decimal_places=3)
    cov = models.DecimalField(max_digits=6, decimal_places=3)
    history = None

    class Meta:
        unique_together = (('cluster', 'protein'),)


class ProteinSequence(Model):
    seq = models.TextField(blank=False)
    length = models.PositiveIntegerField()


class Publication(Model):
    bib_entry = NullCharField(max_length=255, **uniq_opt)
    authors = models.CharField(max_length=1024, **ch_opt)
    title = models.CharField(max_length=512, **ch_opt)
    abstract = models.TextField(**ch_opt)
    keywords = models.CharField(max_length=255, **ch_opt)
    journal = models.CharField(max_length=255, **ch_opt)
    journal_code = models.CharField(max_length=32, **ch_opt)
    doi = NullCharField(max_length=64, **uniq_opt)
    pubmed = NullCharField(max_length=32, **uniq_opt)

    gene = models.ManyToManyField(Gene)
    sample = models.ManyToManyField('Sample')
    study = models.ManyToManyField('Study')

    def clean(self):
        # at least one field must be non-blank
        if not (self.bib_entry or self.doi or self.pubmed):
            raise ValidationError('needs a least one non-blank field')

    def doi_url(self):
        """
        Get URL from DOI
        """
        if self.doi:
            return f'https://doi.org/{self.doi}'
        else:
            raise ValueError('doi is blank')


class Reaction(Model):
    REACTION_SOURCE = (
        (FUNC_DB_KEGG, FUNC_DB_KEGG),
        (FUNC_DB_RHEA, FUNC_DB_RHEA),
        (FUNC_DB_BIOCYC, FUNC_DB_BIOCYC),
    )
    source_accession = models.CharField(
        max_length=32,
        verbose_name='reaction id',
    )
    source_db = models.CharField(
        max_length=32,
        choices=REACTION_SOURCE,
        verbose_name='source database',
    )
    name = models.CharField(
        max_length=255,
        verbose_name='human readable name',
    )

    class Meta:
        unique_together = (
            ('source_accession', 'source_db'),
        )


class ReactionEquivGroup(Model):
    """
    Equivalent reactions group

    Intermediate model for compounds, reactions, and pathways
    """
    # l-kegg-cid = models.ForeignKey(|L-Rhea-cids|L-Biocyc-cids|R-Kegg-cids
    # |R-Rhea-cids|R-Biocyc-cids|Kegg-rxns|Rhea-rxns|Biocyc-rxns|Kegg-pwys
    # |Biocyc-pwys
    reactant = models.ManyToManyField(
        'Compound', related_name='func_with_input',
    )
    product = models.ManyToManyField(
        'Compound', related_name='func_with_output',
    )
    reaction = models.ManyToManyField('Reaction')
    pathway = models.ManyToManyField('Pathway')


class RNA_DB(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='RNA DB id',
    )
    taxon = models.ManyToManyField('Taxonomy')
    lca = models.ForeignKey(
        'Taxonomy', **fk_req,
        related_name='lca_for_rna_db',
    )
    rna_type = models.CharField(max_length=32, verbose_name='type')
    data_source = models.CharField(
        max_length=32, verbose_name='RNAcentral / SILVA reference',
    )


class Sample(Model):
    """
    Biological material taken from an environment, host, or culture

    Multiple "sub-samples" of this may be submitted for sequencing, allowing
    for differtent size fractions to be analysed, and for single sample to be
    analysed via 16S amplicon sequencing, or metagenomic, or transcriptomic
    sequencing, etc.
    """
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='primary sample id',
    )
    study = models.ForeignKey('Study', **fk_req)
    isolation_source = models.CharField(max_length=32)
    collection_date = models.DateField()
    collection_site = models.ForeignKey(
        CollectionSite, **fk_opt,
    )
    sample_depth = models.CharField(max_length=32, **ch_opt)
    ph = models.DecimalField(
        max_digits=3, decimal_places=2, **opt,
        verbose_name='pH',
    )
    temperature = models.DecimalField(
        max_digits=3, decimal_places=1, **opt,
        help_text='temperature in degree Celsius',
    )
    secci = models.DecimalField(max_digits=3, decimal_places=1, **opt)
    partmc = models.DecimalField(
        max_digits=3, decimal_places=1, **opt,
        verbose_name='PartMC',
    )
    dissmc = models.DecimalField(
        max_digits=3, decimal_places=1, **opt,
        verbose_name='DissMC',
    )
    phycocyanin = models.DecimalField(max_digits=3, decimal_places=1, **opt)
    chla = models.DecimalField(max_digits=3, decimal_places=2, **opt)
    doc = models.DecimalField(
        max_digits=3, decimal_places=1, **opt,
        verbose_name='DOC',
    )
    cdom = models.DecimalField(
        max_digits=3, decimal_places=1, **opt,
        verbose_name='CDOM',
    )

    def geo_location(self):
        if self.latitude is None or self.longitude is None:
            return ''

        dir1 = 'N' if self.latitude >= 0 else 'S'  # north is positive
        dir2 = 'E' if self.latitude >= 0 else 'W'  # east is positive
        return f'{abs(self.latitude)} {dir1} {abs(self.longitude)} {dir2}'


class Sequence(Model):
    """
    A DNA sequence

    Uniqueness of sequences stored here is assumed but not enforced
    """
    seq = models.TextField(blank=False)
    length = models.PositiveIntegerField()
    lca = models.ForeignKey('Taxonomy', **fk_opt)


class SequencingProject(Model):
    SEQ_PROJ_TYPES = (
        ('amplicon', 'amplicon'),
        ('metagenome', 'metagenome'),
        ('metatranscriptome', 'metatranscriptome'),
    )
    run = NullCharField(
        max_length=32, **ch_opt, verbose_name='sequencing run id',
    )
    provider = models.CharField(
        max_length=255, verbose_name='sequencing center',
    )
    platform = models.CharField(
        max_length=255, verbose_name='sequencing platform',
    )
    sequencing_type = models.CharField(
        max_length=32, choices=SEQ_PROJ_TYPES,
    )
    library_kit = models.CharField(max_length=32)
    spike_in = models.CharField(
        max_length=64,
        verbose_name='spike-in/standard', **ch_opt,
    )
    sop = models.CharField(
        max_length=255, **ch_opt,
        help_text='description or link to SOP, or similar'
    )
    # amplicon only:
    gene_target = models.CharField(
        max_length=16, **ch_opt,
        help_text='for amplicon sequencing only',
    )
    primers = models.CharField(
        max_length=16, **ch_opt,
        help_text='for amplicon sequencing only',
    )


class Study(Model):
    """
    A study, project, experiment, collection of samples
    """
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='study id',
    )
    sampling_scheme = models.CharField(
        max_length=255, **ch_opt,
    )

    class Meta:
        verbose_name_plural = 'studies'


class Taxonomy(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='NCBI taxonomy ID',
    )
    rank = models.CharField(max_length=32)
    name = models.CharField(max_length=255)
    name_type = models.CharField(max_length=32)
    parent = models.ForeignKey('self', **fk_opt)

    class Meta:
        unique_together = (
            ('rank', 'name', 'name_type'),
        )
        verbose_name_plural = 'taxa'


class UniRef100(Model):
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='UniRef100 accession',
    )
    protein_name = models.CharField(max_length=32, **ch_opt)
    taxonomic_lineage_id = models.CharField(max_length=32, **ch_opt)
    taxonomic_lineage_species = models.CharField(max_length=32, **ch_opt)
    organism = models.CharField(max_length=32, **ch_opt)
    dna_binding = models.CharField(max_length=32, **ch_opt)
    metal_binding = models.CharField(max_length=32, **ch_opt)
    signal_peptide = models.CharField(max_length=32, **ch_opt)
    transmembrane = models.CharField(max_length=32, **ch_opt)
    subcellular_location = models.CharField(max_length=32, **ch_opt)
    tcdb = models.CharField(max_length=32, **ch_opt)
    cog_kog = models.CharField(max_length=32, **ch_opt)
    pfam = models.CharField(max_length=32, **ch_opt)
    tigrfams = models.CharField(max_length=32, **ch_opt)
    kegg = models.CharField(max_length=32, **ch_opt)
    gene_ontology = models.CharField(max_length=32, **ch_opt)
    interpro = models.CharField(max_length=32, **ch_opt)
    ec_number = models.CharField(max_length=32, **ch_opt)
