"""
Module for data load managers
"""
from collections import defaultdict
from functools import wraps
from itertools import groupby, islice, zip_longest
from logging import getLogger
from operator import itemgetter
import os

from django.conf import settings
from django.db.transaction import atomic, set_rollback

from mibios.models import QuerySet
from mibios_umrad.models import TaxID, Taxon, UniRef100
from mibios_umrad.manager import Loader, Manager
from mibios_umrad.utils import (
    CSV_Spec, ProgressPrinter, ReturningGenerator, atomic_dry,
)

from . import get_sample_model, get_dataset_model
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

    @atomic_dry
    def load_sample(self, sample, flag, **kwargs):
        if not kwargs.get('update', False) and getattr(sample, flag):
            raise RuntimeError(f'data already loaded: {flag} -> {sample}')

        if 'file' not in kwargs:
            kwargs.update(file=self.get_file(sample))

        self.load(template={'sample': sample}, **kwargs)

        setattr(sample, flag, True)
        sample.save()


class AlignmentLoader(Loader):
    """ loader XX_GENES.m8 files """

    def get_m8_path(self, sample):
        return sample.get_metagenome_path() / 'annotation' \
            / f'{sample.tracking_id}_GENES.m8'

    def query2gene(self, value, row):
        """
        get gene id from qseqid column for genes

        The query column has all that's in the fasta header, incl. the prodigal
        data.  Also add the sample pk (gene is a FK field).
        """
        gene_id = value.split(maxsplit=1)[0].upper()
        return (self.sample.pk, gene_id)

    def upper(self, value, row):
        """
        upper-case the uniref100 id

        the incoming prefixes have mixed case
        """
        return value.upper()

    spec = CSV_Spec(
        ('gene', query2gene),  # qseqid
        (None, ),  # qlen
        ('ref', upper),  # sseqid
        (None, ),  # slen
        (None, ),  # qstart
        (None, ),  # qend
        (None, ),  # sstart
        (None, ),  # send
        (None, ),  # evalue
        (None, ),  # pident
        (None, ),  # mismatch
        (None, ),  # qcovhsp
        (None, ),  # scovhsp
    )

    @atomic_dry
    def load_m8_sample(self, sample, file=None, **kwargs):
        if sample.gene_alignment_hits_loaded:
            raise RuntimeError(f'alignment data already loaded: {sample}')

        if file is None:
            file = self.get_m8_path(sample)

        self.sample = sample
        self.load(
            spec=self.spec,
            file=file,
            **kwargs,
        )
        # TODO: filter by top parameter
        sample.gene_alignment_hits_loaded = True
        sample.save()


class CompoundAbundanceLoader(Loader, SampleLoadMixin):
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


class SequenceLikeLoader(SampleLoadMixin, Loader):
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

    def get_load_sample_fasta_extra_kw(self, sample):
        """
        Return extra kwargs for from_sample_fasta()

        Should be overwritten by inheriting class if needed
        """
        return {}

    @atomic_dry
    def load_fasta_sample(self, sample, start=0, limit=None, validate=False):
        """
        import sequence data for one sample

        limit - limit to that many contigs, for testing only
        """
        if getattr(sample, self.fasta_load_flag):
            raise RuntimeError(
                'data already loaded - update not supported: '
                '{self.fasta_load_flag} -> {sample}'
            )

        extra = self.get_load_sample_fasta_extra_kw(sample)
        objs = self.from_sample_fasta(sample, start=start, limit=limit,
                                      **extra)
        if validate:
            objs = ((i for i in objs if i.full_clean() or True))
        self.bulk_create(objs)

        setattr(sample, self.fasta_load_flag, True)
        sample.save()

    def from_sample_fasta(self, sample, start=0, limit=None, **extra):
        """
        Generate instances for given sample
        """
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

        self.load_sample(
            sample,
            flag=self.abundance_load_flag,
            spec=self.rpkm_spec,
            file=file,
            update=True,
            **kwargs)

    @atomic
    def XXX_load_sample(self, sample, limit=None, dry_run=False):
        """
        import sequence/coverage data for one sample

        limit - limit to that many contigs, for testing only

        The method assumes that the fasta and coverage input both have the
        contigs/genes in the same order.
        """
        if getattr(sample, self.is_loaded_attr):
            raise RuntimeError(
                f'Sample {sample}: data for {self.model} already loaded'
            )

        extra = self.get_load_sample_fasta_extra_kw(sample)
        objs = self.from_sample_fasta(sample, limit=limit, **extra)
        cov = self.read_coverage(sample, limit=limit)
        cov = ReturningGenerator(cov)
        fk_data, m2m_data = self.read_contigs_file(sample)
        fk_data = self.get_pks_for_rel(fk_data)
        m2m_data = self.get_pks_for_rel(m2m_data)
        objs = self._join_cov_data(objs, cov, fk_data)
        self.bulk_create(objs)
        self.set_m2m_relations(sample, m2m_data['taxid'])
        self.process_coverage_header_data(sample, cov.value)
        setattr(sample, self.is_loaded_attr, True)
        sample.save()

        if dry_run:
            set_rollback(True)

    def read_coverage(self, sample, limit=None):
        """
        load coverage data

        This is a generator, yielding each row.  The return value is a dict
        with the header data.
        """
        rpkm_cols = ['Name', 'Length', 'Bases', 'Coverage', 'Reads', 'RPKM',
                     'Frags', 'FPKM']

        head_data = {}

        with self.get_coverage_path(sample).open('r') as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            print(f'reading {f.name}...')
            # 1. read header
            for line in f:
                if not line.startswith('#'):
                    raise ValueError(
                        '{sample}/{f}: expected header but got: {line}'
                    )
                row = line.strip().lstrip('#').split('\t')
                if row[0] == 'Name':
                    if not row == rpkm_cols:
                        raise ValueError(
                            '{sample}/{f}: unexpected column names'
                        )
                    # table starts
                    break

                # expecting key/value pairs
                key, value = row
                head_data[key] = value

            # 2. read rows
            count = 0
            for line in f:
                if limit and count >= limit:
                    break
                yield line.rstrip().split('\t')
                count += 1

            return head_data

    def _join_cov_data(self, objs, cov, other):
        """
        populate instances with coverage data and other

        other: must be FK data (maps: contig/gene id -> PK)
        """
        null_allowed = {
            field: self.model._meta.get_field(field).null
            for field in other
        }
        skip_count = 0

        # zip_longest: ensures (over just zip) that cov.value gets populated
        for obj, row in zip_longest(objs, cov):
            obj_id = getattr(obj, self.model.id_field_name)
            if obj_id != row[0].split(maxsplit=1)[0].upper():  # check name/id
                raise RuntimeError(
                    f'seq and cov data is out of order: {obj_id=} {row[0]=}'
                )

            obj.length = row[1]
            obj.bases = row[2]
            obj.coverage = row[3]
            obj.reads_mapped = row[4]
            obj.rpkm = row[5]
            obj.frags_mapped = row[6]
            obj.fpkm = row[7]

            # set FKs from contigs file data:
            for fk_field_name, id2value in other.items():
                try:
                    setattr(obj, fk_field_name + '_id', id2value[obj_id])
                except KeyError as e:
                    if null_allowed[fk_field_name]:
                        pass
                    else:
                        # FIXME (error handling)
                        # required FK missing? Comes probably from non-existing
                        # LCA/Taxname/besthit or similar.  Now, in development,
                        # this is to be expected, while source data is based on
                        # a mix of different reference versions, we want to
                        # skip these,later we might want to raise here
                        skip_count += 1
                        break  # skips yield below
                        # alternatively raise:
                        raise RuntimeError(
                            f'{obj=} {fk_field_name=} {obj_id=}'
                        ) from e

            else:
                yield obj

        if skip_count > 0:
            print(f'WARNING: (_join_cov_data): skipped {skip_count} objects')

    def parse_contigs_file(self, sample):
        """
        Get (some) data from contigs txt file for given sample

        Parses and yields data from the contigs files for further processing.
        This also means the contigs file will be read twice, once when
        importing contig data and then again when importing gene data.

        It is assumed that the contigs file is ordered by contig and that
        within the group of a contig's rows, each contig-specific column has
        the same value accross the group's rows.
        """
        cols = ['contig', 'conlen', 'congcnt', 'conpgc', 'conrpkm', 'condepth',
                'gene', 'genepos', 'genepart', 'geneuniq', 'genelen',
                'genepgc', 'generpkm', 'genedepth', 'genetids', 'genelca',
                'contids', 'conlca', 'name', 'maxsco', 'besthit', 'funcs',
                'reac', 'prod', 'tran']

        path = self.get_contigs_file_path(sample)
        with path.open() as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            print(f'reading {f.name}...')
            head = f.readline().rstrip('\n').split('\t')
            if head != cols:
                raise RuntimeError('unexpected header in {f.name}: {head}')

            rows = (line.rstrip('\n').split('\t') for line in f)
            rows = ProgressPrinter(f'{path.name} lines processed')(rows)
            # for groupby assume rows come sorted by contig cluster already
            get_contig_data = itemgetter(0, 1, 2, 3, 4, 5, 16, 17)
            yield from groupby(rows, get_contig_data)

    def get_pks_for_rel(self, data):
        """
        substitute PKs for values for relation data

        Helper method for data loading

        :param data:
            a dict field_name -> field_data which is a dict id->value, this
            is be the output of read_contigs_file()

        This method also fixes some "mixed data" issues with the input data:
            * NCA-* and NOVEL values for LCAs are identified with the QUIDDAM
              lineage (PK == 1)
            * taxid 0 is changed to taxid 1
        """
        if 'taxid' in data:
            taxid2pk = dict(
                TaxID.objects.values_list('taxid', 'pk').iterator()
            )
            data['taxid'] = {
                i: [taxid2pk[j if j else 1] for j in taxids]  # taxid 0 -> 1
                for i, taxids in data['taxid'].items()
            }

        if 'lca' in data:
            str2tax = Taxon.get_parse_and_lookup_fun()
            unknown = defaultdict(list)
            str2pk = {}

            for i, val in data['lca'].items():
                if val.startswith('NCA-') or val == 'NOVEL':
                    # treat as root
                    val = ''
                taxon, missing_key = str2tax(val)
                if taxon is None:
                    unknown[missing_key].append(i)
                else:
                    str2pk[i] = taxon.pk

            if unknown:
                print(f'WARNING: got {len(unknown)} unknown lineages')
                print(f'across {sum((len(i) for i in unknown.values()))} rows')
                for k, v in unknown.items():
                    print('unknown key:', k)
                    print('   ', str(v)[:222])

            if False:
                # DEPRECATED? FIXME
                try:
                    maxpk = Taxon.objects.latest('pk').pk
                except Taxon.DoesNotExist:
                    maxpk = 0
                Taxon.objects.bulk_create(
                    (Taxon.from_name_pks(i) for i in unknown)
                )
                for i in Taxon.objects.filter(pk__gt=maxpk):
                    for j in unknown[i.get_name_pks()]:
                        str2pk[j] = i.pk  # set PK that ws missing earlier
                del maxpk, unknown
            data['lca'] = str2pk
            del str2pk

        # FIXME: the besthit stuff here really belongs into the GeneManager
        # class
        if 'besthit' in data:
            uniref2pk = dict(
                UniRef100.objects.values_list('accession', 'pk').iterator()
            )
            besthit_data = {}
            missing_uniref100 = set()
            for i, j in data['besthit'].items():
                try:
                    besthit_data[i] = uniref2pk[j]
                except KeyError:
                    # unknown uniref100 id
                    missing_uniref100.add(j)

            data['besthit'] = besthit_data
            if missing_uniref100:
                print(
                    f'WARNING: besthit column had {len(missing_uniref100)} '
                    'distinct unknown uniref100 IDs:',
                    ' '.join([i for i in islice(missing_uniref100, 5)])
                )
            del besthit_data, missing_uniref100

        return data

    def set_m2m_relations(self, sample, taxid_m2m_data):
        """
        Set contig-like <-> taxid/taxon relations

        :param dict m2m_data:
            A dict mapping contig/gene ID to lists of TaxID PKs
        """
        obj_id2pk = dict(
            self.filter(sample=sample)
            .values_list(self.model.id_field_name, 'pk')
            .iterator()
        )

        # taxid
        rels = (
            (obj_id2pk[i], j)
            for i, tlist in taxid_m2m_data.items()
            for j in tlist
            if i in obj_id2pk  # to support load_sample()'s limit option
        )
        through = self.model.get_field('taxid').remote_field.through
        us = self.model._meta.model_name + '_id'
        objs = (through(**{us: i, 'taxid_id': j}) for i, j in rels)
        self.bulk_create_wrapper(through.objects.bulk_create)(objs)


class ContigLoader(ContigLikeLoader):
    """ Manager for the Contig model """
    fasta_load_flag = 'contig_fasta_loaded'
    abundance_load_flag = 'contigs_abundance_loaded'

    def get_fasta_path(self, sample):
        return sample.get_metagenome_path() / 'annotation' \
            / (sample.tracking_id + '_MCDD.fa')

    def get_rpkm_path(self, sample):
        return sample.get_metagenome_path() / 'annotation' \
            / f'{sample.tracking_id}_READSvsCONTIGS.rpkm'

    def upper(self, value, record):
        """ upper case contig ids """
        return value.upper()

    rpkm_spec = BBMap_RPKM_Spec(
        ('#Name', 'contig_id', upper),
        ('Length', 'length'),
        ('Bases', 'bases'),
        ('Coverage', 'coverage'),
        ('Reads', 'reads_mapped'),
        ('RPKM', 'rpkm'),
        ('Frags', 'frags_mapped'),
        ('FPKM', 'fpkm'),
    )

    def read_contigs_file(self, sample):
        """
        Get (some) data from contigs txt file for given sample

        This extracts and returns tax ids and LCAs.
        """
        taxid_data = {}
        lca_data = {}
        rows_by_contig = self.parse_contigs_file(sample)

        for (cont_id, *_, contids, conlca), _ in rows_by_contig:
            taxid_data[cont_id] = [int(i) for i in contids.split(';')]
            lca_data[cont_id] = conlca

        return {'lca': lca_data}, {'taxid': taxid_data}

    def process_coverage_header_data(self, sample, data):
        """ Add header data to sample """
        attr_map = {
            # 'File': None,  # TODO: only fwd reads here
            'Reads': 'read_count',
            'Mapped': 'reads_mapped_contigs',
            # 'RefSequences': ??, ignore these, is # of contigs
        }
        for k, attr in attr_map.items():
            setattr(sample, attr, data[k])
            sample.full_clean()
            sample.save()


class FuncAbundanceLoader(Loader, SampleLoadMixin):
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

    def get_rpkm_path(self, sample):
        return sample.get_metagenome_path() / 'annotation' \
            / f'{sample.tracking_id}_READSvsGENES.rpkm'

    def extract_gene_id(self, value, record):
        """ get just the gene id from what was a post-prodigal fasta header """
        return value.split(maxsplit=1)[0].upper()

    rpkm_spec = BBMap_RPKM_Spec(
        ('#Name', 'gene_id', extract_gene_id),
        ('Length', 'length'),
        ('Bases', 'bases'),
        ('Coverage', 'coverage'),
        ('Reads', 'reads_mapped'),
        ('RPKM', 'rpkm'),
        ('Frags', 'frags_mapped'),
        ('FPKM', 'fpkm'),
    )

    @wraps(ContigLikeLoader.load_sample)
    def XXX_load_sample(self, sample, *args, **kwargs):
        if not sample.contigs_ok:
            raise RuntimeError('require to have contigs before loading genes')
        super().load_sample(sample, *args, **kwargs)

    def get_fasta_path(self, sample):
        return (
            sample.get_metagenome_path() / 'annotation'
            / f'{sample.tracking_id}_GENES.fna'
        )

    def get_coverage_path(self, sample):
        return sample.get_metagenome_path() / 'annotation' \
            / f'{sample.tracking_id}_READSvsGENES.rpkm'

    def get_load_sample_fasta_extra_kw(self, sample):
        # returns a dict of the sample's contig
        qs = sample.contig_set.values_list('contig_id', 'pk')
        return dict(contig_ids=dict(qs.iterator()))

    def read_contigs_file(self, sample):
        """
        Get (some) data from contigs txt file for given sample

        This extracts and returns tax ids, LCAs, and besthit
        """
        taxid_data = {}
        lca_data = {}
        besthit_data = {}
        rows_by_contig = self.parse_contigs_file(sample)

        for _, grp in rows_by_contig:
            for row in grp:
                gene_id = row[6]
                taxid_data[gene_id] = [int(i) for i in row[14].split(';')]
                lca_data[gene_id] = row[15]  # should be non-empty
                if row[20]:
                    # besthit may be empty
                    besthit_data[gene_id] = row[20]

        return ({'lca': lca_data, 'besthit': besthit_data},
                {'taxid': taxid_data})

    def process_coverage_header_data(self, sample, data):
        """ Add header data to sample """
        attr_map = {
            # 'File': None,  # TODO: only fwd reads here
            # 'Reads': '',  ignore -- assumed same as for contigs
            'Mapped': 'reads_mapped_genes',
            # 'RefSequences': ??, ignore these, is # of genes
        }
        for k, attr in attr_map.items():
            if k in data:
                setattr(sample, attr, data[k])
                sample.full_clean()
                sample.save()


class SampleManager(Manager):
    """ Manager for the Sample """
    def get_file(self):
        """ get the metagenomic pipeline import log """
        return settings.OMICS_DATA_ROOT / 'data' / 'import_log.tsv'

    @atomic_dry
    def sync(self, source_file=None):
        """
        Update sample table with analysis status


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

                need_save = False
                try:
                    obj = self.get(
                        sample_id=sample_id,
                        dataset__short_name=dataset,
                    )
                except self.model.DoesNotExist:
                    if success != 'TRUE':
                        log.info(f'ignoring {sample_id}: no import success')
                        continue

                    grp_model = get_dataset_model()
                    dataset, new = grp_model.objects.get_or_create(
                        short_name=dataset,
                    )
                    if new:
                        log.info(f'add dataset: {dataset}')

                    obj = self.model(
                        tracking_id=tracking_id,
                        sample_id=sample_id,
                        dataset=dataset,
                        sample_type=sample_type,
                        analysis_dir=analysis_dir,
                    )
                    need_save = True
                    save_info = f'new sample: {obj}'
                else:
                    if success != 'TRUE':
                        log.warning(f'{sample_id}: no import success -- but it'
                                    f'is already in the DB -- skipping')
                        continue

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
                finally:
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


class TaxonAbundanceLoader(Loader, SampleLoadMixin):
    """ loader manager for the TaxonAbundance model """
    ok_field_name = 'tax_abund_ok'

    def get_file(self, sample):
        return resolve_glob(
            sample.get_metagenome_path() / 'annotation',
            f'{sample.tracking_id}_community_*.txt'
        )

    # conversion functions, methods without self, get passed to _load_lines and
    # called ?unbound? or so

    def get_taxname_acc(value, row):
        """ return taxname accession tuple with rank from type column """
        # turn a value "x_n" into (n + 1)
        # (rank is for the "source", so "target" is plus 1)
        letter, _, rank = row[0].partition('_')
        if letter == 'I':
            # FIXME: should be fixed in the source data
            # skip the three rows with ?duplicated? targets
            print(f'WARNING: skipping duplicate with I type: {row[:3]} ...')
            return CSV_Spec.SKIP_ROW
        # items must be in field declaration order, cf. get_accession_lookups()
        try:
            rank = int(rank)
        except ValueError as e:
            raise ValueError(f'{e} in row: {row}') from e
        return (rank + 1, value)

    spec = CSV_Spec(
        ('type', None),  # 1 --> gets picked up via target column
        ('source', None),  # 2
        ('target', 'taxname', get_taxname_acc),  # 3
        ('lin_cnt', 'lin_cnt'),  # 4
        ('lin_avg_prgc', 'lin_avg_prgc'),  # 5
        ('lin_avg_depth', 'lin_avg_depth'),  # 6
        ('lin_avg_rpkm', 'lin_avg_rpkm'),  # 7
        ('lin_gnm_pgc', 'lin_gnm_pgc'),  # 8
        ('lin_sum_sco', 'lin_sum_sco'),  # 9
        ('lin_con_len', 'lin_con_len'),  # 10
        ('lin_gen_len', 'lin_gen_len'),  # 11
        ('lin_con_cnt', 'lin_con_cnt'),  # 12
        ('lin_tgc', 'lin_tgc'),  # 13
        ('lin_comp_genes', 'lin_comp_genes'),  # 14
        ('lin_nlin_gc', 'lin_nlin_gc'),  # 15
        ('lin_novel', 'lin_novel'),  # 16
        ('lin_con_uniq', 'lin_con_uniq'),  # 17
        ('lin_tpg', 'lin_tpg'),  # 18
        ('lin_obg', 'lin_obg'),  # 19
        ('con_lca', 'con_lca'),  # 20
        ('gen_lca', 'gen_lca'),  # 21
        ('part_gen', 'part_gen'),  # 22
        ('uniq_gen', 'uniq_gen'),  # 23
        ('con_len', 'con_len'),  # 24
        ('gen_len', 'gen_len'),  # 25
        ('con_rpkm', 'con_rpkm'),  # 26
        ('gen_rpkm', 'gen_rpkm'),  # 27
        ('gen_dept', 'gen_dept'),  # 28
    )
