"""
Module for data load managers
"""
from collections import defaultdict
from functools import wraps
from itertools import chain, groupby, islice, zip_longest
from logging import getLogger
from operator import itemgetter
import os

from django.conf import settings
from django.db.transaction import atomic, set_rollback

from mibios.models import QuerySet
from mibios_umrad.models import Lineage, TaxName, Taxon, UniRef100
from mibios_umrad.manager import Loader, Manager
from mibios_umrad.utils import CSV_Spec, ProgressPrinter, ReturningGenerator

from . import get_sample_model
from .utils import get_fasta_sequence


log = getLogger(__name__)


class SampleLoadMixin:
    """ Mixin for Loader class that loads per-sample files """
    ok_field_name = None

    @atomic
    def load_sample(self, sample, start=0, max_rows=None, dry_run=False):
        if self.ok_field_name is not None:
            if getattr(sample, self.ok_field_name):
                raise RuntimeError(f'data already loaded: {sample}')

        self.load(
            file=self.get_file(sample),
            template={'sample': sample},
            start=start,
            max_rows=max_rows,
        )

        if self.ok_field_name is not None:
            setattr(sample, self.ok_field_name, True)
            sample.save()

        set_rollback(dry_run)


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
        return settings.OMICS_DATA_ROOT / 'GENES' \
            / f'{sample.accession}_compounds_{settings.OMICS_DATA_VERSION}.txt'


class SequenceLikeQuerySet(QuerySet):
    """ objects manager for sequence-like models """

    def to_fasta(self):
        """
        Make fasta-formatted sequences
        """
        files = {}
        lines = []
        fields = ('seq_offset', 'seq_bytes', 'gene_id', 'sample__accession')
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


class SequenceLikeLoader(SequenceLikeManager):
    """ Loader manager for the SequenceLike abstract model """

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

    @atomic
    def load_sample(self, sample, limit=None, verbose=False, dry_run=False):
        """
        import sequence data for one sample

        limit - limit to that many contigs, for testing only
        """
        extra = self.get_load_sample_fasta_extra_kw(sample)
        objs = self.from_sample_fasta(sample, limit=limit, verbose=verbose,
                                      **extra)
        self.bulk_create(objs)
        set_rollback(dry_run)

    def from_sample_fasta(self, sample, limit=None, **extra):
        """
        Generate instances for given sample
        """
        with self.get_fasta_path(sample).open('r') as fa:
            os.posix_fadvise(fa.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            print(f'reading {fa.name} ...')
            obj = None
            count = 0
            pos = 0
            eof = ''  # end of file
            for line in chain(fa, [eof]):
                if limit and count >= limit:
                    break
                pos += len(line)
                if line == eof or line.startswith('>'):
                    if obj is None:
                        pass  # first line
                    else:
                        obj.seq_bytes = pos - obj.seq_offset
                        # obj.full_clean()  # way too slow
                        yield obj
                        count += 1

                    if line == eof:
                        break

                    # make next obj
                    obj = self.model(
                        sample=sample,
                        seq_offset=pos,
                    )
                    try:
                        obj.set_from_fa_head(line, **extra)
                    except Exception as e:
                        raise RuntimeError(
                            f'failed parsing fa head in file {fa.name}: '
                            f'{e.__class__.__name__}: {e}, line:\n{line}'
                        ) from e


class ContigLikeLoader(SequenceLikeLoader):
    """ Manager for ContigLike abstract model """
    is_loaded_attr = None  # set in inheriting class

    def get_contigs_file_path(self, sample):
        return settings.OMICS_DATA_ROOT / 'GENES' \
            / f'{sample.accession}_contigs_{settings.OMICS_DATA_VERSION}.txt'

    def process_coverage_header_data(self, sample, data):
        """ Add header data to sample """
        # must be implemented by inheriting class
        raise NotImplementedError

    @atomic
    def load_sample(self, sample, limit=None, dry_run=False):
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
        self.set_taxon_m2m_relations(sample, m2m_data['taxon'])
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
        if 'taxon' in data:
            taxid2pk = dict(
                Taxon.objects.values_list('taxid', 'pk').iterator()
            )
            data['taxon'] = {
                i: [taxid2pk[j if j else 1] for j in taxids]  # taxid 0 -> 1
                for i, taxids in data['taxon'].items()
            }

        if 'lca' in data:
            str2lin = Lineage.get_parse_and_lookup_fun()
            root_lin = str(Taxon.objects.get(taxid=1).lineage)
            new = defaultdict(list)
            str2pk = {}
            missing_taxnames = set()
            for i, val in data['lca'].items():
                if val.startswith('NCA-') or val == 'NOVEL':
                    # FIXME: what to do with these?
                    val = root_lin
                try:
                    lin, missing_key = str2lin(val)
                except TaxName.DoesNotExist as e:
                    missing_taxnames.add(e.args[0])  # add (name, rankid) tuple
                    continue
                if lin is None:
                    new[missing_key].append(i)
                else:
                    str2pk[i] = lin.pk

            if missing_taxnames:
                print(f'WARNING: got {len(missing_taxnames)} unknown taxnames '
                      'in lca column:',
                      ' '.join([str(i) for i in islice(missing_taxnames, 5)]))
            del missing_taxnames

            # make missing LCA lineages
            if new:
                try:
                    maxpk = Lineage.objects.latest('pk').pk
                except Lineage.DoesNotExist:
                    maxpk = 0
                Lineage.objects.bulk_create(
                    (Lineage.from_name_pks(i) for i in new)
                )
                for i in Lineage.objects.filter(pk__gt=maxpk):
                    for j in new[i.get_name_pks()]:
                        str2pk[j] = i.pk  # set PK that ws missing earlier
                del maxpk, new
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

    def set_taxon_m2m_relations(self, sample, taxon_m2m_data):
        """
        Set contig-like <-> taxon relations

        :param dict taxon_m2m_data:
            A dict mapping contig/gene ID to a list of Taxon PKs
        """
        thing_id2pk = dict(
            self.filter(sample=sample)
            .values_list(self.model.id_field_name, 'pk')
            .iterator()
        )
        rels = (
            (thing_id2pk[i], j)
            for i, tlist in taxon_m2m_data.items()
            for j in tlist
            if i in thing_id2pk  # to support load_sample()'s limit option
        )
        through = self.model.get_field('taxon').remote_field.through
        us = self.model._meta.model_name + '_id'
        objs = (through(**{us: i, 'taxon_id': j}) for i, j in rels)
        self.bulk_create_wrapper(through.objects.bulk_create)(objs)


class ContigClusterLoader(ContigLikeLoader):
    """ Manager for the ContigCluster model """
    is_loaded_attr = 'contigs_ok'

    def get_fasta_path(self, sample):
        return settings.OMICS_DATA_ROOT / 'ASSEMBLIES' / 'MERGED' \
            / (sample.accession + '_MCDD.fa')

    def get_coverage_path(self, sample):
        return settings.OMICS_DATA_ROOT / 'ASSEMBLIES' / 'COVERAGE' \
            / f'{sample.accession}_READSvsCONTIGS.rpkm'

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

        return {'lca': lca_data}, {'taxon': taxid_data}

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
        return settings.OMICS_DATA_ROOT / 'GENES' \
            / f'{sample.accession}_functions_{settings.OMICS_DATA_VERSION}.txt'

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
    is_loaded_attr = 'genes_ok'

    @wraps(ContigLikeLoader.load_sample)
    def load_sample(self, sample, *args, **kwargs):
        if not sample.contigs_ok:
            raise RuntimeError('require to have contigs before loading genes')
        super().load_sample(sample, *args, **kwargs)

    def get_fasta_path(self, sample):
        return (
            settings.OMICS_DATA_ROOT / 'GENES'
            / (sample.accession + '_GENES.fna')
        )

    def get_coverage_path(self, sample):
        return (settings.OMICS_DATA_ROOT / 'GENES' / 'COVERAGE'
                / f'{sample.accession}_READSvsGENES.rpkm')

    def get_load_sample_fasta_extra_kw(self, sample):
        # returns a dict of the sample's contig clusters
        qs = sample.contigcluster_set.values_list('cluster_id', 'pk')
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
                {'taxon': taxid_data})

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
        return settings.OMICS_DATA_ROOT / 'sample_list.txt'

    @atomic
    def sync(self, source_file=None, dry_run=False, **kwargs):
        if source_file is None:
            source_file = self.get_file()

        with open(source_file) as f:
            seen = []
            for line in f:
                obj, isnew = self.get_or_create(
                    accession=line.strip()
                )
                seen.append(obj.pk)
                if isnew:
                    log.info(f'new sample: {obj}')

        not_in_src = self.exclude(pk__in=seen)
        if not_in_src.exists():
            log.warning(f'Have {not_in_src.count()} extra samples in DB not '
                        f'found in {source_file}')
        set_rollback(dry_run)

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
                i.contigcluster_set.count(),
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
        return settings.OMICS_DATA_ROOT / 'GENES' \
            / f'{sample.accession}_community_{settings.OMICS_DATA_VERSION}.txt'

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
