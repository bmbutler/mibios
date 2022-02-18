"""
Module for data load managers
"""
from collections import defaultdict
from functools import wraps
from itertools import chain, groupby, zip_longest
from operator import itemgetter
import os

from django.conf import settings
from django.db.transaction import atomic, set_rollback

from mibios_umrad.models import Lineage, Taxon, UniRef100
from mibios_umrad.manager import BaseManager
from mibios_umrad.utils import ProgressPrinter, ReturningGenerator


class SequenceLikeLoader(BaseManager):
    """ Manager for the SequenceLike abstrasct model """

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
                        raise RuntimeError(
                            f'{obj=} {fk_field_name=} {obj_id=}'
                        ) from e

            yield obj

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
            for i, val in data['lca'].items():
                if val.startswith('NCA-') or val == 'NOVEL':
                    # FIXME: what to do with these?
                    val = root_lin
                lin, missing_key = str2lin(val)
                if lin is None:
                    new[missing_key].append(i)
                else:
                    str2pk[i] = lin.pk

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
            data['besthit'] = {
                i: uniref2pk[j]
                for i, j in data['besthit'].items()
            }
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
        BaseManager.bulk_create_wrapper(through.objects.bulk_create)(objs)


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
