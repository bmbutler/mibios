"""
Module for data load managers

=== Workflow to load metagenomic data ===

Assumes that UMRAD and sample meta-data is loaded
    Sample.objects.sync()
    s = Sample.objects.get(sample_id='samp_14')
    Contig.loader.load_fasta_sample(s)
    Gene.loader.load_fasta_sample(s)
    Contig.loader.load_abundance_sample(s, bulk=False)
    Gene.loader.load_abundance_sample(s, bulk=False)
    Alignment.loader.load_sample(s)
    Gene.loader.assign_gene_lca(s)
    Contig.loader.assign_gene_lca(s)
    TaxonAbundance.loader.load_sample(s)

"""
from itertools import groupby, islice
from logging import getLogger
from operator import itemgetter
import os

from django.conf import settings
from django.db.models import prefetch_related_objects
from django.utils.module_loading import import_string

from mibios.models import QuerySet
from mibios.umrad.models import TaxID, Taxon, UniRef100
from mibios.umrad.manager import BulkLoader, Manager
from mibios.umrad.utils import CSV_Spec, atomic_dry

from . import get_sample_model
from .utils import get_fasta_sequence


log = getLogger(__name__)


# FIXME move function to good home
def resolve_glob(path, pat):
    value = None
    for i in path.glob(pat):
        if value is None:
            return i
        else:
            raise RuntimeError(f'glob is not unique: {i}')
    raise FileNotFoundError(f'glob does not resolve: {path / pat}')


class BBMap_RPKM_Spec(CSV_Spec):
    def process_header(self, file):
        # skip initial non-rows
        pos = file.tell()
        while True:
            line = file.readline()
            if line.startswith('#'):
                if line.startswith('#Name'):
                    file.seek(pos)
                    return super().process_header(file)
                else:
                    # TODO: process rpkm header data
                    pos = file.tell()
            else:
                raise RuntimeError(
                    f'rpkm header lines must start with #, column header with '
                    f'#Name, got: {line}'
                )


class SampleLoadMixin:
    """ Mixin for Loader class that loads per-sample files """

    load_flag_attr = None
    """ may be specified by implementing class """

    sample = None
    """ sample is set by load_sample() for use in per-field helper methods """

    @atomic_dry
    def load_sample(self, sample, **kwargs):
        if 'flag' in kwargs:
            flag = kwargs.pop('flag')
            if flag is None:
                # explicit override / no flag check/set
                pass
        else:
            flag = self.load_flag_attr

        if flag and not kwargs.get('update', False) and getattr(sample, flag):
            raise RuntimeError(f'data already loaded: {flag} -> {sample}')

        if 'file' not in kwargs:
            kwargs.update(file=self.get_file(sample))

        self.sample = sample
        self.load(template={'sample': sample}, **kwargs)
        # ensure subsequent calls of manager methods never get wrong sample:
        self.sample = None

        if flag:
            setattr(sample, flag, True)
            sample.save()


class M8Spec(CSV_Spec):
    def iterrows(self):
        """
        Transform m8 file into top good hits

        Yields rows of triplets [gene, uniref100, score]

        m8 file format:

             0 qseqid  /  $gene
             1 qlen
             2 sseqid  / $hit
             3 slen
             4 qstart
             5 qend
             6 sstart
             7 send
             8 evalue
             9 pident / $pid
            10 mismatch
            11 qcovhsp  /  $cov
            12 scovhsp
        """
        minid = self.loader.MINID
        mincov = self.loader.MINCOV
        score_margin = self.loader.TOP
        per_gene = groupby(super().iterrows(), key=itemgetter(0))
        for gene_id, grp in per_gene:
            # 1. get good hits
            hits = []
            for row in grp:
                pid = float(row[9])
                if pid < minid:
                    continue
                cov = float(row[11])
                if cov < mincov:
                    continue
                # hits: (gene, uniref100, score)
                hits.append((row[0], row[2], int(pid * cov)))

            # 2. keep top hits
            if hits:
                hits = sorted(hits, key=lambda x: -x[2])  # sort by score
                minscore = int(hits[0][2] * score_margin)
                for i in hits:
                    # The int() above are improper rounding, so some scores a
                    # bit below the cutoff will slip through
                    if i[2] >= minscore:
                        yield i


class AlignmentLoader(BulkLoader, SampleLoadMixin):
    """
    loads data from XX_GENES.m8 files into Alignment table

    Parameters:
      1. min query coverage (60%)
      2. min pct alignment identity (60%)
      3. min number of genes to keep (3) (not including? phages/viruses)
      4. top score margin (0.9)
      5. min number of model genomes for func. models (5)

    Notes on Annotatecontigs.pl:
        top == 0.9
        minimum pident == 60%
        minimum cov/qcovhsp == 60
        score = pid*cov
        keep top score per hit (global?)
        keep top score of hits for each gene
        keep top x percentile hits for each gene


    """
    MINID = 60.0
    MINCOV = 60.0
    TOP = 0.9

    load_flag_attr = 'gene_alignment_hits_loaded'

    def get_file(self, sample):
        return sample.get_metagenome_path() / f'{sample.tracking_id}_GENES.m8'

    def query2gene_pk(self, value, row, obj):
        """
        get gene PK from qseqid column for genes

        The query column has all that's in the fasta header, incl. the prodigal
        data.  Also add the sample pk (gene is a FK field).
        """
        gene_id = value.split('\t', maxsplit=1)[0].partition('_')[2]
        return self.gene_id_map[gene_id]

    def upper(self, value, row, obj):
        """
        upper-case the uniref100 id

        the incoming prefixes have mixed case
        """
        return value.upper()

    spec = M8Spec(
        ('gene.pk', query2gene_pk),
        ('ref', upper),
        ('score',),
    )

    def load_sample(self, sample, file=None, **kwargs):
        self.gene_id_map = dict(sample.gene_set.values_list('gene_id', 'pk'))
        if file is None:
            file = self.get_file(sample)
        super().load(file=file, **kwargs)


class CompoundAbundanceLoader(BulkLoader, SampleLoadMixin):
    """ loader manager for CompoundAbundance """
    ok_field_name = 'comp_abund_ok'

    spec = CSV_Spec(
        ('cpd', 'compound'),
        ('cpd_name', None),
        ('type', 'role'),
        ('CPD_GENES', None),
        ('CPD_SCO', 'scos'),
        ('CPD_RPKM', 'rpkm'),
        ('cpdlca', None),
    )

    def get_file(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.tracking_id}_compounds_*.txt'
        )


class SequenceLikeQuerySet(QuerySet):
    """ objects manager for sequence-like models """

    def to_fasta(self):
        """
        Make fasta-formatted sequences
        """
        files = {}
        lines = []
        fields = ('fasta_offset', 'fasta_len', 'gene_id', 'sample__accession')
        qs = self.select_related('sample').values_list(*fields)
        try:
            for offs, length, gene_id, sampid in qs.iterator():
                if sampid not in files:
                    sample = get_sample_model().objects.get(accession=sampid)
                    files[sampid] = \
                        self.model.loader.get_fasta_path(sample).open('rb')

                lines.append(f'>{sampid}:{gene_id}')
                lines.append(
                    get_fasta_sequence(files[sampid], offs, length).decode()
                )
        finally:
            for i in files.values():
                i.close()

        return '\n'.join(lines)


SequenceLikeManager = Manager.from_queryset(SequenceLikeQuerySet)


class SequenceLikeLoader(SampleLoadMixin, BulkLoader):
    """
    Loader manager for the SequenceLike abstract model

    Provides the load_fasta_sample() method.
    """

    def get_fasta_path(self, sample):
        """
        return path to fasta file that contains our sequence
        """
        # must be implemented by inheriting class
        # should return Path
        raise NotImplementedError

    def get_set_from_fa_head_extra_kw(self, sample):
        """
        Return extra kwargs for from_sample_fasta()

        Should be overwritten by inheriting class if needed
        """
        return {}

    @atomic_dry
    def load_fasta_sample(self, sample, start=0, limit=None, bulk=True,
                          validate=False):
        """
        import sequence data for one sample

        limit - limit to that many contigs, for testing only
        """
        if getattr(sample, self.fasta_load_flag):
            raise RuntimeError(
                f'data already loaded - update not supported: '
                f'{self.fasta_load_flag} -> {sample}'
            )

        objs = self.from_sample_fasta(sample, start=start, limit=limit)
        if validate:
            objs = ((i for i in objs if i.full_clean() or True))

        if bulk:
            self.bulk_create(objs)
        else:
            for i in objs:
                i.save()

        setattr(sample, self.fasta_load_flag, True)
        sample.save()

    def from_sample_fasta(self, sample, start=0, limit=None):
        """
        Generate instances for given sample

        Helper for load_fasta_sample().
        """
        extra = self.get_set_from_fa_head_extra_kw(sample)
        with self.get_fasta_path(sample).open('r') as fa:
            os.posix_fadvise(fa.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            print(f'reading {fa.name} ...')
            obj = None
            data = self._fasta_parser(fa)
            for header, offs, byte_len in islice(data, start, limit):
                obj = self.model(
                    sample=sample,
                    fasta_offset=offs,
                    fasta_len=byte_len,
                )
                try:
                    obj.set_from_fa_head(header, **extra)
                except Exception as e:
                    raise RuntimeError(
                        f'failed parsing fa head in file {fa.name}: '
                        f'{e.__class__.__name__}: {e}:{header}'
                    ) from e
                yield obj

    def _fasta_parser(self, file):
        """ helper to iterate over fasta record infos """
        header = None
        record_offs = None
        while True:
            pos = file.tell()
            line = file.readline()
            if line.startswith('>') or not line:
                if header is not None:
                    yield header, record_offs, pos - record_offs
                header = line.lstrip('>').rstrip()
                record_offs = pos

            if not line:
                # EOF
                break


class ContigLikeLoader(SequenceLikeLoader):
    """ Manager for ContigLike abstract model """
    abundance_load_flag = None  # set in inheriting class

    def get_contigs_file_path(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.tracking_id}_contigs_*.txt'
        )

    def process_coverage_header_data(self, sample, data):
        """ Add header data to sample """
        # must be implemented by inheriting class
        raise NotImplementedError

    rpkm_spec = None
    """ rpkm_spec must be set in inheriting classes """

    @atomic_dry
    def load_abundance_sample(self, sample, file=None, **kwargs):
        """
        Load data from bbmap *.rpkm files

        Assumes that fasta data was loaded previously.
        """
        if file is None:
            file = self.get_rpkm_path(sample)

        # read header data
        reads = None
        mapped = None
        with open(file) as ifile:
            print(f'Reading from {ifile.name} ...')
            while True:
                pos = ifile.tell()
                line = ifile.readline()
                if line.startswith('#Name\t') or not line.startswith('#'):
                    # we're past the preamble
                    ifile.seek(pos)
                    break
                key, _, data = line.strip().partition('\t')
                if key == '#Reads':
                    reads = int(data)
                elif key == '#Mapped':
                    mapped = int(data)

        if reads is None or mapped is None:
            raise RuntimeError('Failed parsing (mapped) read counts')

        if sample.read_count is not None and sample.read_count != reads:
            print(f'Warning: overwriting existing read count with'
                  f'different value: {sample.read_count}->{reads}')
        sample.read_count = reads

        mapped_old_val = getattr(sample, self.reads_mapped_sample_attr)
        if mapped_old_val is not None and mapped_old_val != mapped:
            print(f'Warning: overwriting existing mapped read count with'
                  f'different value: {mapped_old_val}->{mapped}')
        setattr(sample, self.reads_mapped_sample_attr, mapped)
        sample.save()

        self.load_sample(
            sample,
            flag=self.abundance_load_flag,
            spec=self.rpkm_spec,
            file=file,
            update=True,
            **kwargs)

    @staticmethod
    def get_lca(taxa):
        """ helper to calculate the LCA Taxon from taxids """
        lins = []
        for i in taxa:
            # assume ancestors are prefetched, so sort by python
            lineage = sorted(i.ancestors.all(), key=lambda x: x.rank)
            lineage.append(i)
            lins.append(lineage)

        lca = None
        for items in zip(*lins):
            if len(set([i.pk for i in items])) == 1:
                lca = items[0]
            else:
                break

        return lca


class ContigLoader(ContigLikeLoader):
    """ Manager for the Contig model """
    fasta_load_flag = 'contig_fasta_loaded'
    abundance_load_flag = 'contigs_abundance_loaded'
    reads_mapped_sample_attr = 'reads_mapped_contigs'

    def get_fasta_path(self, sample):
        return sample.get_metagenome_path() / 'assembly' \
            / (sample.tracking_id + '_MCDD.fa')

    def get_rpkm_path(self, sample):
        return sample.get_metagenome_path() / 'assembly' \
            / f'{sample.tracking_id}_READSvsCONTIGS.rpkm'

    def trim_id(self, value, row, obj):
        """ trim tracking id off, e.g. deadbeef_123 => 123 """
        _, _, value = value.partition('_')
        return value.upper()

    def calc_rpkm(self, value, row, obj):
        """ calculate rpkm based on total post-QC read-pairs """
        return (1_000_000_000 * int(obj.reads_mapped)
                / int(obj.length) / self.sample.read_count)

    def calc_fpkm(self, value, row, obj):
        """ calculate fpkm based on total post-QC read-pairs """
        return (1_000_000_000 * int(obj.frags_mapped)
                / int(obj.length) / self.sample.read_count)

    rpkm_spec = BBMap_RPKM_Spec(
        ('#Name', 'contig_id', trim_id),
        ('Length', 'length'),
        ('Bases', 'bases'),
        ('Coverage', 'coverage'),
        ('Reads', 'reads_mapped'),
        ('RPKM', 'rpkm_bbmap'),
        ('Frags', 'frags_mapped'),
        ('FPKM', 'fpkm_bbmap'),
        (BBMap_RPKM_Spec.CALC_VALUE, 'rpkm', calc_rpkm),
        (BBMap_RPKM_Spec.CALC_VALUE, 'fpkm', calc_fpkm),
    )

    @atomic_dry
    def assign_contig_lca(self, sample):
        """
        assign / pre-compute taxids and LCAs to contigs via genes

        This populates the Contig.lca/taxid fields
        """
        Gene = import_string('mibios.omics.models.Gene')
        # TODO: make this a Contig manager method?  Or Sample method?
        genes = Gene.objects.filter(sample=sample)
        # genes = genes.exclude(hits=None)
        genes = genes.values_list('contig_id', 'pk', 'lca_id')
        genes = genes.order_by('contig_id')  # _id is contig's PK here
        print('Fetching taxonomy... ', end='', flush=True)
        taxa = Taxon.objects.filter(gene__sample=sample).distinct().in_bulk()
        print(f'{len(taxa)} [OK]')
        print('Fetching contigs... ', end='', flush=True)
        contigs = self.model.objects.filter(sample=sample).in_bulk()
        print(f'{len(contigs)} [OK]')
        print('Fetching Gene -> TaxIDs links... ', end='', flush=True)
        g2tids_qs = Gene._meta.get_field('taxid').remote_field.through.objects
        g2tids_qs = g2tids_qs.filter(gene__sample=sample)
        g2tids_qs = g2tids_qs.values_list('gene_id', 'taxid_id')
        g2tids_qs = groupby(g2tids_qs.order_by('gene_id'), key=lambda x: x[0])
        g2tids = {}
        for gene_pk, grp in g2tids_qs:
            g2tids[gene_pk] = [i for _, i in grp]
        print(f'{len(g2tids)} [OK]')

        def contig_taxid_links():
            """ generate m2m links with side-effects """
            for contig_pk, grp in groupby(genes, lambda x: x[0]):
                taxid_pks = set()
                lcas = set()
                for _, gene_pk, lca_pk in grp:
                    try:
                        taxid_pks.update(g2tids[gene_pk])
                        lcas.add(taxa[lca_pk])
                    except KeyError:
                        # gene w/o hits
                        pass

                # assign contig LCA from gene LCAs (side-effect):
                contigs[contig_pk].lca = self.get_lca(lcas)

                for i in taxid_pks:
                    yield (contig_pk, i)

        Through = self.model._meta.get_field('taxid').remote_field.through
        delcount, _ = Through.objects.filter(contig__sample=sample).delete()
        if delcount:
            print(f'Deleted {delcount} existing contig-taxid links')
        objs = (
            Through(contig_id=i, taxid_id=j)
            for i, j in contig_taxid_links()
        )
        self.bulk_create_wrapper(Through.objects.bulk_create)(objs)

        self.bulk_update(contigs.values(), fields=['lca'])


class FuncAbundanceLoader(BulkLoader, SampleLoadMixin):
    ok_field_name = 'func_abund_ok'

    def get_file(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.tracking_id}_functionss_*.txt'
        )

    spec = CSV_Spec(
        ('fid', 'function'),
        ('fname', None),
        ('FUNC_GENES', None),
        ('FUNC_SCOS', 'scos'),
        ('FUNC_RPKM', 'rpkm'),
        ('fidlca', None),
    )


class GeneLoader(ContigLikeLoader):
    """ Manager for the Gene model """
    fasta_load_flag = 'gene_fasta_loaded'
    abundance_load_flag = 'contigs_abundance_loaded'
    reads_mapped_sample_attr = 'reads_mapped_genes'

    def get_fasta_path(self, sample):
        return (
            sample.get_metagenome_path() / 'genes'
            / f'{sample.tracking_id}_GENES.fna'
        )

    def get_rpkm_path(self, sample):
        return sample.get_metagenome_path() / 'genes' \
            / f'{sample.tracking_id}_READSvsGENES.rpkm'

    def extract_gene_id(self, value, row, obj):
        """ get just the gene id from what was a post-prodigal fasta header """
        # deadbeef_123_1 # bla bla bla => 123_1
        return value.split(maxsplit=1)[0].partition('_')[2]

    def calc_rpkm(self, value, row, obj):
        """ calculate rpkm based on total post-QC read-pairs """
        return (1_000_000_000 * obj.reads_mapped
                / obj.length / self.sample.read_count)

    def calc_fpkm(self, value, row, obj):
        """ calculate fpkm based on total post-QC read-pairs """
        return (1_000_000_000 * obj.frags_mapped
                / obj.length / self.sample.read_count)

    rpkm_spec = BBMap_RPKM_Spec(
        ('#Name', 'gene_id', extract_gene_id),
        ('Length', 'length'),
        ('Bases', 'bases'),
        ('Coverage', 'coverage'),
        ('Reads', 'reads_mapped'),
        ('RPKM', 'rpkm_bbmap'),
        ('Frags', 'frags_mapped'),
        ('FPKM', 'fpkm_bbmap'),
        (BBMap_RPKM_Spec.CALC_VALUE, 'rpkm', calc_rpkm),
        (BBMap_RPKM_Spec.CALC_VALUE, 'fpkm', calc_fpkm),
    )

    def get_set_from_fa_head_extra_kw(self, sample):
        # returns a dict of the sample's contigs
        qs = sample.contig_set.values_list('contig_id', 'pk')
        return dict(contig_id_map=dict(qs.iterator()))

    @atomic_dry
    def assign_gene_lca(self, sample):
        """
        assign / pre-compute taxids and LCAs to genes via uniref100 hits

        This also populates Gene.lca / Gene.besthit fields
        """
        Alignment = import_string('mibios.omics.models.Alignment')

        # Get UniRef100.taxids m2m links
        TUT = UniRef100._meta.get_field('taxids').remote_field.through
        qs = TUT.objects.filter(uniref100__gene_hit__sample=sample).distinct()
        qs = qs.order_by('uniref100').values_list('uniref100_id', 'taxid_id')
        print('Fetching uniref100/taxids...', end='', flush=True)
        u2t = {}  # pk map ur100->taxids
        for ur100_pk, grp in groupby(qs.iterator(), key=lambda x: x[0]):
            u2t[ur100_pk] = [i for _, i in grp]
        print(f'{len(u2t)} [OK]')

        print('Fetching tax info...', end='', flush=True)
        qs = (
            TaxID.objects
            .filter(classified_uniref100__gene_hit__sample=sample)
            .select_related('taxon')
        )
        taxmap = {i.pk: i.taxon for i in qs.iterator()}
        print(f'{len(taxmap)} [OK]')

        print('Fetching ancestry...', end='', flush=True)
        taxa = list(taxmap.values())
        prefetch_related_objects(taxa, 'ancestors')
        print(f'{len(taxa)} [OK]')

        hits = Alignment.objects.filter(gene__sample=sample)
        hits = hits.select_related('gene').order_by('gene')
        genes = []

        def gene_taxid_links():
            """
            generate pk pairs (gene, taxid) to make Gene<->TaxID m2m links

            As a side-effect this also sets besthit and lca for each gene that
            has a hit.
            """
            for gene, grp in groupby(hits.iterator(), key=lambda x: x.gene):
                # get all TaxIDs for all hits, some UR100 may not have taxids
                tids = set()
                best = None
                for align in grp:
                    if best is None or align.score > best.score:
                        best = align
                    for taxid_pk in u2t.get(align.ref_id, []):
                        if taxid_pk in tids:
                            continue
                        else:
                            tids.add(taxid_pk)
                            yield (gene.pk, taxid_pk)
                gene.besthit_id = best.ref_id
                gene.lca = self.get_lca([taxmap[i] for i in tids])
                # some genes here have hits but no taxids, for those lca is set
                # to None here
                genes.append(gene)

        Through = self.model._meta.get_field('taxid').remote_field.through
        delcount, _ = Through.objects.filter(gene__sample=sample).delete()
        if delcount:
            print(f'Deleted {delcount} existing gene-taxid links')
        objs = (Through(gene_id=i, taxid_id=j) for i, j in gene_taxid_links())
        self.bulk_create_wrapper(Through.objects.bulk_create)(objs)

        # update lca field for all genes incl. the unknowns
        self.bulk_update(genes, fields=['lca', 'besthit_id'])
        print('Erasing lca for genes without any hits... ', end='', flush=True)
        num_unknown = self.filter(hits=None).update(lca=None)
        print(f'{num_unknown} [OK]')


class SampleManager(Manager):
    """ Manager for the Sample """
    def get_file(self):
        """ get the metagenomic pipeline import log """
        return settings.OMICS_DATA_ROOT / 'data' / 'import_log.tsv'

    @atomic_dry
    def sync(self, source_file=None, create=False, skip_on_error=False):
        """
        Update sample table with analysis status

        :param bool create:
            Create a new sample if needed.  The new sample will not belong to a
            dataset.  Not for production use.
        """
        if source_file is None:
            source_file = self.get_file()

        with open(source_file) as f:
            # check assumptions on columns
            SAMPLE_ID = 0
            PROJECT = 1
            TYPE = 2
            TRACKING_ID = 7
            ANALYSIS_DIR = 12
            SUCCESS = 16
            cols = (
                (SAMPLE_ID, 'Sample'),
                (PROJECT, 'Project'),
                (TYPE, 'sample_type'),
                (TRACKING_ID, 'sampleID'),
                (ANALYSIS_DIR, 'sample_dir'),
                (SUCCESS, 'import_sucess'),  # [sic]
            )

            head = f.readline().rstrip('\n').split('\t')
            for index, column in cols:
                if head[index] != column:
                    raise RuntimeError(
                        f'unexpected header in {f.name}: 0-based column '
                        f'{index} is {head[index]} but {column} is expected'
                    )

            seen = []
            track_id_seen = set()
            unchanged = 0
            for line in f:
                row = line.rstrip('\n').split('\t')
                sample_id = row[SAMPLE_ID]
                dataset = row[PROJECT]
                sample_type = row[TYPE]
                tracking_id = row[TRACKING_ID]
                analysis_dir = row[ANALYSIS_DIR]
                success = row[SUCCESS]

                if not all([sample_id, tracking_id, analysis_dir, success]):
                    raise RuntimeError(f'field is empty in row: {row}')

                if tracking_id in track_id_seen:
                    # skip rev line
                    continue
                else:
                    track_id_seen.add(tracking_id)

                if success != 'TRUE':
                    log.info(f'ignoring {sample_id}: no import success')
                    continue

                need_save = False
                try:
                    obj = self.get(
                        sample_id=sample_id,
                        # dataset__short_name=dataset,
                    )
                except self.model.DoesNotExist:
                    if not create:
                        if skip_on_error:
                            log.warning(f'no such sample: {sample_id} / '
                                        f'{dataset} (skipping)')
                            continue
                        else:
                            raise

                    # create a new orphan sample
                    obj = self.model(
                        tracking_id=tracking_id,
                        sample_id=sample_id,
                        dataset=None,
                        sample_type=sample_type,
                        analysis_dir=analysis_dir,
                    )
                    need_save = True
                    save_info = f'new sample: {obj}'
                else:
                    updateable = ['tracking_id', 'sample_type', 'analysis_dir']
                    changed = []
                    for attr in updateable:
                        val = locals()[attr]
                        if getattr(obj, attr) != val:
                            setattr(obj, attr, val)
                            changed.append(attr)
                    if changed:
                        need_save = True
                        save_info = (f'update: {obj} changed: '
                                     f'{", ".join(changed)}')
                    else:
                        unchanged += 1

                if need_save:
                    obj.metag_pipeline_reg = True
                    obj.full_clean()
                    obj.save()
                    log.info(save_info)

                seen.append(obj.pk)

        if unchanged:
            log.info(f'Data for {unchanged} samples are already in the DB and '
                     f'remain unchanged')

        not_in_src = self.exclude(pk__in=seen)
        if not_in_src.exists():
            log.warning(f'Have {not_in_src.count()} extra samples in DB not '
                        f'found in {source_file}')

    def status(self):
        if not self.exists():
            print('no samples in database yet')
            return

        print(' ' * 10, 'contigs', 'bins', 'checkm', 'genes', sep='\t')
        for i in self.all():
            print(
                f'{i}:',
                'OK' if i.contigs_ok else '',
                'OK' if i.binning_ok else '',
                'OK' if i.checkm_ok else '',
                'OK' if i.genes_ok else '',
                sep='\t'
            )

    def status_long(self):
        if not self.exists():
            print('no samples in database yet')
            return

        print(' ' * 10, 'cont cl', 'MAX', 'MET93', 'MET97', 'MET99', 'genes',
              sep='\t')
        for i in self.all():
            print(
                f'{i}:',
                i.contig_set.count(),
                i.binmax_set.count(),
                i.binmet93_set.count(),
                i.binmet97_set.count(),
                i.binmet99_set.count(),
                i.gene_set.count(),
                sep='\t'
            )


class TaxonAbundanceLoader(Manager):
    """ loader manager for the TaxonAbundance model """
    ok_field_name = 'tax_abund_ok'

    @atomic_dry
    def load_sample(self, sample, validate=False):
        """
        populate the taxon abundance table for a single sample

        This requires that the sample's genes' LCAs have been set.
        """
        print('Fetching taxonomy... ', end='', flush=True)
        Ancestry = Taxon._meta.get_field('ancestors').remote_field.through
        qs = (
            Ancestry.objects
            .filter(from_taxon__gene__sample=sample)
            .distinct()
            .values_list('from_taxon_id', 'to_taxon_id')
            .order_by('from_taxon_id')
        )
        # map of PKs from a gene's taxon to list of ancestor tax nodes
        # (for those that have ancestors, so not root's immediate children
        # (domains of life and unplaced stuff))
        ancestors = {
            from_pk: [i for _, i in grp]
            for from_pk, grp in groupby(qs, key=lambda x: x[0])
        }
        print(f'{len(ancestors)} [OK]')

        print('Accumulating RPKMs... ', end='', flush=True)
        data = {}
        qs = sample.gene_set.values_list('lca_id', 'rpkm')
        for lca_id, gene_rpkm in qs.iterator():
            if lca_id is None:
                continue

            for j in [lca_id] + ancestors.get(lca_id, []):
                try:
                    data[j] += gene_rpkm
                except KeyError:
                    data[j] = gene_rpkm
        print(f'{len(data)} [OK]')

        objs = (
            self.model(sample=sample, taxon_id=k, sum_gene_rpkm=v)
            for k, v in data.items()
        )

        if validate:
            objs = list(objs)
            for i in objs:
                try:
                    i.full_clean()
                except Exception:
                    print(f'validation failed: {vars(i)}')
                    raise

        self.bulk_create(objs)
        setattr(sample, self.ok_field_name, True)
        sample.save()
