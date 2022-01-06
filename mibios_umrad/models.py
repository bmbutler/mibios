from collections import OrderedDict
from itertools import islice, zip_longest
from logging import getLogger
import os
from pathlib import Path

from django.conf import settings
from django.db import models
from django.db.transaction import atomic
from django.utils.functional import cached_property

from mibios import get_registry
from mibios.models import Model

from .fields import AccessionField
from .model_utils import (
    ch_opt, fk_opt, VocabularyModel, delete_all_objects_quickly,
)
from .utils import ProgressPrinter, chunker


log = getLogger(__name__)


class DryRunRollback(Exception):
    pass


class Biocyc(Model):
    history = None
    name = models.CharField(max_length=64, unique=True)

    class Meta:
        verbose_name = 'BioCyc'
        verbose_name_plural = verbose_name


class COG(Model):
    history = None
    accession = AccessionField()

    class Meta:
        verbose_name = 'COG'
        verbose_name_plural = 'COGs'

    def __str__(self):
        return self.accession


class Compound(Model):
    """ Chemical compound, reactant, or product """
    history = None
    name = models.CharField(max_length=32, unique=True)  # can be CHEBI id


class EC(Model):
    history = None
    accession = AccessionField()

    class Meta:
        verbose_name = 'EC'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Function(Model):
    history = None
    name = models.CharField(max_length=128, unique=True)


class GeneOntology(Model):
    history = None
    accession = AccessionField(prefix='GO:')

    class Meta:
        verbose_name = 'GeneOntology'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Interpro(Model):
    history = None
    accession = AccessionField(prefix='IPR')

    class Meta:
        verbose_name = 'Interpro'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class KEGG(Model):
    history = None
    accession = AccessionField(prefix='R')

    class Meta:
        verbose_name = 'KEGG'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Location(VocabularyModel):
    pass

    class Meta:
        verbose_name = 'subcellular location'


class Metal(VocabularyModel):
    pass


class PFAM(Model):
    history = None
    accession = AccessionField(prefix='PF')

    class Meta:
        verbose_name = 'PFAM'

    def __str__(self):
        return self.accession


class RHEA(Model):
    history = None
    accession = AccessionField()  # numbers

    class Meta:
        verbose_name = 'RHEA'
        verbose_name_plural = 'RHEA'

    def __str__(self):
        return self.accession


class TaxName(Model):
    history = None

    RANKS = (
        (0, 'root'),
        (1, 'domain'),
        (2, 'phylum'),
        (3, 'class', 'klass'),
        (4, 'order'),
        (5, 'family'),
        (6, 'genus'),
        (7, 'species'),
        (8, 'strain'),
    )
    RANK_CHOICE = ((i[0], i[1]) for i in RANKS)

    rank = models.PositiveSmallIntegerField(choices=RANK_CHOICE)
    name = models.CharField(max_length=64)

    class Meta:
        unique_together = (('rank', 'name'),)
        verbose_name = 'taxonomic name'

    def __str__(self):
        return f'{self.get_rank_display()} {self.name}'

    @classmethod
    @atomic
    def load(cls, path=None):
        if path is None:
            path = cls.get_taxonomy_path()

        data = []
        with path.open() as f:
            pp = ProgressPrinter('taxa found')
            log.info(f'reading taxonomy: {path}')
            for line in f:
                data.append(line.strip().split('\t'))
                pp.inc()

            pp.finish()

        pp = ProgressPrinter('tax names processed')
        rankids = [i[0] for i in TaxName.RANKS[1:]]
        names = dict()
        for row in data:
            for rid, name in zip_longest(rankids, row[1:]):
                if rid is None:
                    raise RuntimeError(f'unexpectedly low ranks: {row}')
                if name is None:
                    # no strain etc
                    continue
                key = (rid, name)
                pp.inc()
                if key not in names:
                    names[key] = TaxName(rank=rid, name=name)

        pp.finish()

        log.info(f'Storing {len(names)} unique tax names to DB...')
        TaxName.objects.bulk_create(names.values())

        return data

    @classmethod
    def get_taxonomy_path(cls):
        return settings.UMRAD_ROOT / 'TAXON' / 'TAXONOMY_DB_DEC_2021.txt'


class Taxon(Model):
    history = None

    taxid = models.PositiveIntegerField(unique=True)
    domain = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_dom_rel',
    )
    phylum = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_phy_rel',
    )
    klass = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_cls_rel',
    )
    order = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_ord_rel',
    )
    family = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_fam_rel',
    )
    genus = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_gen_rel',
    )
    species = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_sp_rel',
    )
    strain = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_str_rel',
    )

    class Meta:
        verbose_name_plural = 'taxa'

    def __str__(self):
        return f'{self.taxid} {self.lineage}'

    def get_lineage(self):
        lineage = []
        for i in TaxName.RANKS[1:]:
            attr = i[-1]
            name = getattr(self, attr, None)
            if name is None:
                break
            else:
                lineage.append(name.name)
        return lineage

    lineage_list = cached_property(get_lineage, name='lineage_list')

    def names(self, with_missing_ranks=True):
        """ return dict of taxnames """
        names = OrderedDict()
        for i in TaxName.RANKS[1:]:
            attr = i[-1]
            name = getattr(self, attr, None)
            if name is None and not with_missing_ranks:
                break
            else:
                names[attr] = name

        return names

    @classmethod
    def format_lineage(cls, lineage, sep=';'):
        """
        Format a list of str taxnames as lineage
        """
        return sep.join(lineage)

    @property
    def lineage(self):
        return self.format_lineage(self.lineage_list)

    @classmethod
    def lca_lineage(cls, taxa):
        """
        Return lineage of LCA of given taxa

        Arguments:
            taxa: Taxon queryset or iterable of taxids or Taxon instances
        """
        if not taxa:
            raise ValueError(
                'taxa should be a list or iterable with at least one element'
            )
        # ranks are rank field attr names str from 'domain' to 'strain'
        ranks = [j[-1] for j in TaxName.RANKS[1:]]

        # lca: a list of tuples (rank_name, taxname_id)
        lca = None
        for i in taxa:
            if isinstance(i, int):
                obj = cls.objects(taxid=i)
            else:
                obj = i

            if lca is None:
                # init lca
                lca = []
                for r in ranks:
                    rid = getattr(obj, r + '_id', None)
                    if rid is None:
                        break
                    lca.append((r, rid))
                continue

            # calc new lca
            new_lca = []
            for r, rid in lca:
                if rid == getattr(obj, r + '_id', None):
                    new_lca.append((r, rid))
                    continue
                else:
                    break
            lca = new_lca

        # retrieve names
        qs = TaxName.objects.filter(pk__in=[i[1] for i in lca])
        names = dict(qs.values_list('pk', 'name'))

        # return just the tax names
        return [names[i[1]] for i in lca]

    @classmethod
    @atomic
    def load(cls, path=None):
        data = TaxName.load(path)

        # reloading names to get the ids, depends on order the fields are
        # declared
        names = {
            (rank, name): pk for pk, rank, name
            in TaxName.objects.values_list().iterator()
        }

        pp = ProgressPrinter('taxa processed')
        objs = []
        # ranks: get pairs of rank id and rank field attribute name
        ranks = [(i[0], i[-1]) for i in TaxName.RANKS[1:]]
        for row in data:
            taxid = row[0]
            kwargs = dict(taxid=taxid)
            for (rid, attr), name in zip_longest(ranks, row[1:]):
                # we should always have rid, attr here since we went through
                # data before, name may be None for missing low ranks
                if name is not None:
                    # assign ids directly
                    kwargs[attr + '_id'] = names[(rid, name)]

            objs.append(cls(**kwargs))
            pp.inc()

        pp.finish()
        log.info(f'Storing {len(objs)} taxa in DB...')
        cls.objects.bulk_create(objs)

    @classmethod
    def classified(cls, lineage):
        """ remove unclassified tail of a lineage """
        ranks = [i[1].upper() for i in cls.RANK_CHOICE[1:]]
        ret = lineage[:1]  # keep first
        for rank, name in zip(ranks, lineage[1:]):
            if name == f'UNCLASSIFIED_{ret[-1]}_{rank}':
                break
            ret.append(name)

        return ret


class TIGR(Model):
    history = None
    accession = AccessionField(prefix='TIGR')

    class Meta:
        verbose_name = 'TIGR'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Uniprot(Model):
    history = None
    accession = AccessionField(verbose_name='uniprot id')

    class Meta:
        verbose_name = 'Uniprot'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class UniRef100(Model):
    history = None

    # The fields below are based on the columns in UNIREF100_INFO_DEC_2021.txt
    # in order.  To ensure update_m2m() works, the related models of all m2m
    # fields are assumed to have the unique key used in the source table as the
    # first unique, non-pk field

    #  0 UNIREF100
    accession = AccessionField(prefix='UNIREF100_')
    #  1 NAME
    function = models.ManyToManyField(Function)
    #  2 LENGTH
    length = models.PositiveIntegerField(blank=True, null=True)
    #  3 UNIPROT_IDS
    uniprot = models.ManyToManyField(Uniprot)
    #  4 UNIREF90
    uniref90 = AccessionField(prefix='UNIREF90_', unique=False)
    #  5 TAXON_IDS
    taxon = models.ManyToManyField(Taxon)
    #  6 LINEAGE (method)
    #  7 SIGALPEP
    signal_peptide = models.CharField(max_length=32, **ch_opt)
    #  8 TMS
    tms = models.CharField(max_length=128, **ch_opt)
    #  9 DNA
    dna_binding = models.CharField(max_length=128, **ch_opt)
    # 10 METAL
    metal_binding = models.ManyToManyField(Metal)
    # 11 TCDB
    tcdb = models.CharField(max_length=128, **ch_opt)
    # 12 LOCATION
    subcellular_location = models.ManyToManyField(Location)
    # 13 COG
    cog_kog = models.ManyToManyField(COG)
    # 14 PFAM
    pfam = models.ManyToManyField(PFAM)
    # 15 TIGR
    tigr = models.ManyToManyField(TIGR)
    # 16 GO
    gene_ontology = models.ManyToManyField(GeneOntology)
    # 17 IPR
    interpro = models.ManyToManyField(Interpro)
    # 18 EC
    ec = models.ManyToManyField(EC)
    # 19 KEGG
    kegg = models.ManyToManyField(KEGG)
    # 20 RHEA
    rhea = models.ManyToManyField(RHEA)
    # 21 BIOCYC
    biocyc = models.ManyToManyField(Biocyc)
    # 22 REACTANTS
    reactant = models.ManyToManyField(Compound, related_name='reactant_of')
    # 23 PRODUCTS
    product = models.ManyToManyField(Compound, related_name='product_of')
    # 24 TRANS_CPD
    trans_cpd = models.ManyToManyField(Compound, related_name='trans_of')

    class Meta:
        verbose_name = 'UniRef100'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    def lineage(self):
        return ';'.join(Taxon.lcs_lineage(self.taxon_set.all()))

    @classmethod
    def get_file(cls):
        return (Path(settings.UMRAD_ROOT) / 'UNIPROT'
                / 'UNIREF100_INFO_DEC_2021.txt')

    @classmethod
    def load_file(cls, max_rows=None, dry_run=True):
        cols = (
            'UNIREF100', 'NAME', 'LENGTH', 'UNIPROT_IDS', 'UNIREF90',
            'TAXON_IDS', 'LINEAGE', 'SIGALPEP', 'TMS', 'DNA', 'METAL', 'TCDB',
            'LOCATION', 'COG', 'PFAM', 'TIGR', 'GO', 'IPR', 'EC', 'KEGG',
            'RHEA', 'BIOCYC', 'REACTANTS', 'PRODUCTS', 'TRANS_CPD'
        )
        with cls.get_file().open() as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            first = f.readline().strip().split('\t')
            for i, (a, b) in enumerate(zip(first, cols), start=1):
                if a != b:
                    raise ValueError(
                        f'Unexpected header row: first mispatch in column {i}:'
                        f'got "{a}", expected "{b}"'
                    )

            if max_rows is None:
                file_it = f
            else:
                file_it = islice(f, max_rows)

            try:
                return cls.load_lines(file_it, dry_run=dry_run)
            except DryRunRollback:
                print('[dry run rollback]')

    @classmethod
    @atomic
    def load_lines(cls, lines, dry_run=True):
        objs = []
        m2m_data = {}
        split = cls._split_m2m_input
        pp = ProgressPrinter('UniRef100 records read')
        for line in lines:
            obj = cls()
            m2m = {}
            row = line.rstrip('\n').split('\t')

            if len(row) != 25:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {pp.current=} {row=}'
                )

            obj.accession = row[0]
            m2m['function'] = split(row[1])
            obj.length = int(row[2]) if row[2] else None
            m2m['uniprot'] = split(row[3])
            obj.uniref90 = row[4]
            m2m['taxon'] = split(row[5], int)
            obj.signal_peptide = row[7]
            obj.tms = row[8]
            obj.dna_binding = row[9]
            m2m['metal_binding'] = split(row[10])
            obj.tcdb = row[11]
            m2m['subcellular_location'] = split(row[12])
            m2m['cog_kog'] = split(row[13])
            m2m['pfam'] = split(row[14])
            m2m['tigr'] = split(row[15])
            m2m['gene_ontology'] = split(row[16])
            m2m['interpro'] = split(row[17])
            m2m['ec'] = split(row[18])
            m2m['kegg'] = split(row[19])
            m2m['rhea'] = split(row[20])
            m2m['biocyc'] = split(row[21])
            m2m['reactant'] = split(row[22])
            m2m['product'] = split(row[23])
            m2m['trans_cpd'] = split(row[24])

            objs.append(obj)
            m2m_data[obj.accession] = m2m
            pp.inc()

        pp.finish()
        if not objs:
            # empty file?
            return

        m2m_fields = list(m2m.keys())
        del m2m

        batch_size = 990  # sqlite3 max is around 999 ?
        pp = ProgressPrinter('UniRef100 records written to DB')
        for batch in chunker(objs, batch_size):
            try:
                batch = cls.objects.bulk_create(batch)
            except Exception:
                print(f'ERROR saving UniRef100: batch 1st: {vars(batch[0])=}')
                raise
            pp.inc(len(batch))
            if batch[0].pk is not None:
                print(batch[0], batch[0].pk)

        del objs
        pp.finish()

        # get accession -> pk map
        accpk = dict(cls.objects.values_list('accession', 'pk').iterator())

        # replace accession with pk in m2m data keys
        m2m_data = {accpk[i]: data for i, data in m2m_data.items()}
        del accpk

        # collecting all m2m entries
        for i in m2m_fields:
            cls.update_m2m(i, m2m_data)

        if dry_run:
            raise DryRunRollback

    @classmethod
    def update_m2m(cls, field_name, m2m_data):
        """
        Update M2M data for one field

        :param str field_name: Name of m2m field
        :param dict m2m_data:
            A dict with all fields' m2m data as produced in the load_lines
            method.
        """
        print(f'm2m {field_name}: ', end='', flush=True)
        field = cls._meta.get_field(field_name)
        model = field.related_model
        # get the first unique, non-pk field (and be hopeful)
        rel_key_field = [
            i for i in model._meta.get_fields()
            if hasattr(i, 'unique') and i.unique and not i.primary_key
        ][0]

        # extract and flatten all keys for field in m2m data
        keys = {i for objdat in m2m_data.values() for i in objdat[field_name]}
        print(f'{len(keys)} unique keys in data - ', end='', flush=True)
        if not keys:
            print()
            return

        # get existing
        qs = model.objects.all()
        qs = qs.values_list(rel_key_field.name, 'pk')
        old = dict(qs.iterator())
        print(f'known: {len(old)} ', end='', flush=True)

        # save new
        new_keys = [i for i in keys if i not in old]
        print(f'new: {len(new_keys)} ', end='', flush=True)
        if new_keys:
            if field_name == 'taxon':
                print()
                raise RuntimeError(
                    f'no auto-adding new {field_name} entries: '
                    + ' '.join([f'"{i}"' for i in new_keys[:20]])
                    + '...'
                )
            new_related = (model(**{rel_key_field.name: i}) for i in new_keys)
            new_related = model.objects.bulk_create(new_related)
            print('(saved)', end='', flush=True)

        # get m2m field's key -> pk mapping
        accpk = dict(
            model.objects.values_list(rel_key_field.name, 'pk').iterator()
        )

        # set relationships
        rels = []  # pairs of UniRef100 and Uniprot PKs
        for i, other in m2m_data.items():
            rels.extend(((i, accpk[j]) for j in other[field_name]))
        Through = field.remote_field.through  # the intermediate model
        fk_id_name = model._meta.model_name + '_id'
        through_objs = [
            Through(
                **{'uniref100_id': i, fk_id_name: j}
            )
            for i, j in rels
        ]
        Through.objects.bulk_create(through_objs)

        print(f' ({len(through_objs)} relations saved)', end='', flush=True)
        print()

    @classmethod
    def _split_m2m_input(cls, value, type_conv=lambda x: x):
        """
        Helper to split semi-colon-separated list-field values in import file
        """
        # filters for '' from empty values
        items = [type_conv(i) for i in value.split(';') if i]
        # TODO: duplicates in input data (NAME/function column), tell Teal?
        items = list(set(items))
        return items


# development stuff
def delete_all_uniref100_etc():
    r = get_registry()
    for i in r.apps['mibios_umrad'].get_models():
        if i._meta.model_name.startswith('tax'):
            continue
        print(f'Deleting: {i} ', end='', flush=True)
        delete_all_objects_quickly(i)
        print('[done]')
