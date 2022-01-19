from collections import OrderedDict
from itertools import groupby, islice, zip_longest
from logging import getLogger
from operator import itemgetter
import os
from pathlib import Path

from django.conf import settings
from django.db import models
from django.db.transaction import atomic
from django.utils.functional import cached_property

from mibios import get_registry


from .fields import AccessionField
from .model_utils import (
    ch_opt, fk_opt, fk_req, VocabularyModel, delete_all_objects_quickly,
    LoadMixin, Model,
)
from .utils import ProgressPrinter, chunker, DryRunRollback


log = getLogger(__name__)


class COG(Model):
    accession = AccessionField()

    class Meta:
        verbose_name = 'COG'
        verbose_name_plural = 'COGs'

    def __str__(self):
        return self.accession


class CompoundEntry(Model, LoadMixin):
    """ Reference DB's entry for chemical compound, reactant, or product """
    DB_BIOCYC = 'b'
    DB_CHEBI = 'c'
    DB_HMDB = 'h'
    DB_INCHI = 'i'
    DB_KEGG = 'k'
    DB_PUBCHEM = 'p'
    DB_CHOICES = (
        ('Biocyc', DB_BIOCYC),
        ('ChEBI', DB_CHEBI),
        ('HMDB', DB_HMDB),
        ('InChI', DB_INCHI),
        ('KEGG', DB_KEGG),
        ('PubChem', DB_PUBCHEM),
    )

    accession = models.CharField(max_length=40, unique=True)
    db = models.CharField(max_length=1, choices=DB_CHOICES, db_index=True)
    formula = models.CharField(max_length=32, blank=True)
    charge = models.SmallIntegerField(blank=True, null=True)
    mass = models.CharField(max_length=16, blank=True)  # TODO: decimal??
    name = models.ManyToManyField('CompoundName')
    compound = models.ForeignKey('Compound', **fk_req)

    def __str__(self):
        return self.accession


class Compound(Model):
    """ Unique source-DB-independent compounds """

    @classmethod
    def get_file(cls):
        return Path(
            '/geomicro/data22/teals_pipeline/BOSS/REFDB2_COMPOUNDS_REACTIONS'
            '/all_compound_info_SEP_2021.txt'
        )

    # spec: 2nd item is either base compound field name or compound model name
    # for the reverse relation
    import_file_spec = (
        ('id', 'accession'),
        ('form', 'formula'),
        ('char', 'charge'),
        ('mass', 'mass'),
        ('hmdb', CompoundEntry.DB_HMDB),
        ('inch', CompoundEntry.DB_INCHI),
        ('bioc', CompoundEntry.DB_BIOCYC),
        ('kegg', CompoundEntry.DB_KEGG),
        ('pubc', CompoundEntry.DB_PUBCHEM),
        ('cheb', CompoundEntry.DB_CHEBI),
        ('name', 'name'),
    )

    @classmethod
    @atomic
    def load(cls, max_rows=None, dry_run=True):
        refdb_names = [
            i for _, i in CompoundEntry.DB_CHOICES
            if i in [j[1] for j in cls.import_file_spec]
        ]

        # get data and split m2m fields
        data = []
        for row in super().load(max_rows=max_rows, parse_only=True):
            for i in refdb_names + ['name']:
                row[i] = cls._split_m2m_input(row[i])
            data.append(row)

        unicomps = []  # collector for unique compounds (this model)
        name_data = {}  # accession/id to names association

        grpk = itemgetter(*refdb_names)  # sort/group by related IDs columns
        data = sorted(data, key=grpk)
        pp = ProgressPrinter('compound records processed')
        data = pp(data)
        extra_ids_count = 0
        for key, grp in groupby(data, grpk):
            grp = list(grp)
            all_ids = {i for j in key for i in j}  # flatten list-of-list
            cgroup = []

            if len(grp) < len(all_ids):
                # NOTE: there are extra IDs in the xref columns for which we
                # don't have a row, they will not enter the compound group, and
                # get lost here
                # print(f'WARNING: skipping for group size inconsistency: '
                #       f'{len(grp)=} {len(all_ids)=} {key=}')

                # TODO: make this raise when fixed in the source dataa?
                # raise RuntimeError(
                #     f'expected to have one group member per unique ID, but: '
                #     f'{key=} {grp=}'
                # )
                extra_ids_count += 1

            elif len(grp) > len(all_ids):
                # this case has not happend yet
                raise RuntimeError(
                    f'more group members ({len(grp)=}) than IDs '
                    f'({len(all_ids)})? {key=} {grp=}'
                )

            for row in grp:
                acc = row['accession']

                # get the ref db for this row:
                for refdb, ids in zip(refdb_names, key):
                    if acc in ids:
                        break
                else:
                    raise RuntimeError(
                        f'accession {acc} not found in other ID fields: {grp}'
                    )
                del ids

                comp_obj = CompoundEntry(
                    accession=acc,
                    db=refdb,
                    formula=row['formula'],
                    charge=None if row['charge'] == '' else row['charge'],
                    mass=row['mass']
                )
                cgroup.append(comp_obj)
                name_data[acc] = row['name']

            unicomps.append(cgroup)

        if extra_ids_count:
            print(f'WARNING: {extra_ids_count} compound groups with (ignored) '
                  f'extra IDs')

        # create Compound objects and get PKs
        objs = (Compound() for _ in range(len(unicomps)))
        cls.objects.bulk_create(objs, batch_size=500)
        pks = cls.objects.values_list('pk', flat=True)

        # cross-link compound entries (and re-pack to one list)
        comps = []
        for pk, group in zip(pks, unicomps):
            for i in group:
                i.compound_id = pk
                comps.append(i)
        del pk, unicomps

        # store compound entries
        CompoundEntry.objects.bulk_create(comps)
        del comps

        # store names
        uniq_names = set()
        for items in name_data.values():
            uniq_names.update(items)

        name_objs = (CompoundName(entry=i) for i in uniq_names)
        CompoundName.objects.bulk_create(name_objs)

        # read back names with PKs
        name_pk_map = dict(
            CompoundName.objects.values_list('entry', 'pk').iterator()
        )

        # Set name relations
        pk_map = dict(
            CompoundEntry.objects.values_list('accession', 'pk').iterator()
        )
        # comp->name relation:
        rels = (
            (comp_pk, name_pk_map[name_entry])
            for acc, comp_pk in pk_map.items()
            for name_entry in name_data[acc]
        )
        through = CompoundEntry._meta.get_field('name').remote_field.through
        through_objs = [
            through(
                **{'compoundentry_id': i, 'compoundname_id': j}
            )
            for i, j in rels
        ]
        pp = ProgressPrinter('compound entry vs. name relations')
        through_objs = pp(through_objs)
        through.objects.bulk_create(through_objs)


class CompoundName(VocabularyModel):
    max_length = 128


class EC(Model):
    accession = AccessionField()

    class Meta:
        verbose_name = 'EC'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Function(Model):
    name = models.CharField(max_length=128, unique=True)


class GeneOntology(Model):
    accession = AccessionField(prefix='GO:')

    class Meta:
        verbose_name = 'GeneOntology'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Interpro(Model):
    accession = AccessionField(prefix='IPR')

    class Meta:
        verbose_name = 'Interpro'
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
    accession = AccessionField(prefix='PF')

    class Meta:
        verbose_name = 'PFAM'

    def __str__(self):
        return self.accession


class ReactionEntry(Model):
    DB_BIOCYC = 'b'
    DB_KEGG = 'k'
    DB_RHEA = 'r'
    DB_CHOICES = (
        ('Biocyc', DB_BIOCYC),
        ('KEGG', DB_KEGG),
        ('RHEA', DB_RHEA),
    )

    accession = AccessionField()
    db = models.CharField(max_length=1, choices=DB_CHOICES, db_index=True)
    bi_directional = models.BooleanField()
    left = models.ManyToManyField(
        CompoundEntry, related_name='to_reaction',
    )
    right = models.ManyToManyField(
        CompoundEntry, related_name='from_reaction',
    )
    reaction = models.ForeignKey('Reaction', **fk_req)

    def __str__(self):
        return self.accession


class Reaction(Model):
    """ unique DB-independent reaction """

    import_file_spec = (
        ('ID', 'accession'),
        ('dir', 'dir'),
        ('left_kegg', 'left_kegg'),
        ('left_biocyc', 'left_biocyc'),
        ('left_rhea', 'left_rhea'),
        ('right_kegg', 'right_kegg'),
        ('right_biocyc', 'right_biocyc'),
        ('right_rhea', 'right_rhea'),
        ('kegg_rxn', ReactionEntry.DB_KEGG),
        ('biocyc_rxn', ReactionEntry.DB_BIOCYC),
        ('rhea_rxn', ReactionEntry.DB_RHEA),
    )

    @classmethod
    def get_file(cls):
        return Path(
            '/geomicro/data22/teals_pipeline/BOSS/REFDB2_COMPOUNDS_REACTIONS/'
            'all_reaction_info_SEP_2021.txt'
        )

    @classmethod
    @atomic
    def load(cls, max_rows=None, dry_run=True):
        refdbs = [
            i for _, i in ReactionEntry.DB_CHOICES
            if i in [j[1] for j in cls.import_file_spec]
        ]
        comp_cols = {
            ReactionEntry.DB_KEGG: ('left_kegg', 'right_kegg'),
            ReactionEntry.DB_BIOCYC: ('left_biocyc', 'right_biocyc'),
            ReactionEntry.DB_RHEA: ('left_rhea', 'right_rhea'),
        }

        # get data and split m2m fields
        data = []
        for row in super().load(max_rows=max_rows, parse_only=True):
            m2mcols = [i for _, i in cls.import_file_spec if i not in ['accession', 'dir']]  # noqa:E501
            for i in m2mcols:
                row[i] = cls._split_m2m_input(row[i])
            data.append(row)

        urxns = []

        # sort/group and process by reaction group
        grpk = itemgetter(*refdbs)
        data = sorted(data, key=grpk)
        pp = ProgressPrinter('reaction entry records processed')
        data = pp(data)
        extra_ids_count = 0
        compounds = {}
        for key, grp in groupby(data, grpk):
            grp = list(grp)
            all_ids = {i for j in key for i in j}  # flatten list-of-list
            rxngroup = []

            if len(grp) < len(all_ids):
                # NOTE: see this check in Compound
                extra_ids_count += 1

            elif len(grp) > len(all_ids):
                # this case has not happend yet ?
                raise RuntimeError(
                    f'more group members ({len(grp)=}) than IDs '
                    f'({len(all_ids)})? {key=} {grp=}'
                )

            for row in grp:
                acc = row['accession']

                # get the ref db for this row:
                for refdb, ids in zip(refdbs, key):
                    if acc in ids:
                        break
                else:
                    raise RuntimeError(
                        f'accession {acc} not found in other ID fields: {grp}'
                    )
                del ids

                rxn_obj = ReactionEntry(
                    accession=acc,
                    db=refdb,
                    bi_directional=True if row['dir'] == 'BOTH' else False
                )
                rxngroup.append(rxn_obj)

                for i, (left_col, right_col) in comp_cols.items():
                    if i == refdb:
                        compounds[acc] = refdb, row[left_col], row[right_col]
                        break
                    # TODO: account for what we miss here?
                else:
                    raise RuntimeError('logic bug: no match for db key')

            urxns.append(rxngroup)

        del key, grp, row, acc, rxn_obj, left_col, right_col

        if extra_ids_count:
            print(f'WARNING: {extra_ids_count} reaction groups with (ignored) '
                  f'extra IDs')

        # create Reaction objects and get PKs
        objs = (cls() for _ in range(len(urxns)))
        cls.objects.bulk_create(objs, batch_size=500)
        pks = cls.objects.values_list('pk', flat=True)

        # cross-link reaction entries (and re-pack to one list)
        rxns = []
        for pk, group in zip(pks, urxns):
            for i in group:
                i.reaction_id = pk
                rxns.append(i)
        del pk, group, urxns

        # store reaction entries
        ReactionEntry.objects.bulk_create(rxns)
        del rxns

        # deal with unknown compounds
        qs = CompoundEntry.objects.values_list('accession', flat=True)
        known_cpd_accs = set(qs.iterator())
        del qs
        unknown_cpd_accs = {}
        for rxndb, left, right in compounds.values():
            for i in left + right:
                if i in known_cpd_accs:
                    continue
                if i in unknown_cpd_accs:
                    # just check dbkey
                    if rxndb != unknown_cpd_accs[i]:
                        raise RuntimeError('inconsistent db key: {i=} {dbkey=}'
                                           '{unknown_cpd_accs[i]=}')
                else:
                    # add
                    unknown_cpd_accs[i] = rxndb
        del rxndb, left, right, known_cpd_accs
        if unknown_cpd_accs:
            print(f'Found {len(unknown_cpd_accs)} unknown compound IDs in '
                  'reaction data!')
            max_pk = Compound.objects.order_by('pk').last().pk
            unicomp_objs = (Compound() for _ in range(len(unknown_cpd_accs)))
            Compound.objects.bulk_create(unicomp_objs, batch_size=500)
            uni_pks = Compound.objects.filter(pk__gt=max_pk)\
                              .values_list('pk', flat=True)
            CompoundEntry.objects.bulk_create((
                CompoundEntry(
                    accession=acc,
                    db=CompoundEntry.DB_CHEBI if rxndb == ReactionEntry.DB_RHEA else rxndb,  # noqa: E501
                    compound_id=pk
                )
                for (acc, rxndb), pk in zip(unknown_cpd_accs.items(), uni_pks)
            ))
            del uni_pks, max_pk
        del unknown_cpd_accs

        # get reaction entry accession to PK mapping (with db info)
        rxn_qs = ReactionEntry.objects.values_list('accession', 'db', 'pk')

        # get compound acc->pk mapping (with db info)
        qs = CompoundEntry.objects.values_list('accession', 'db', 'pk')
        comp_acc2pk = {acc: (db, pk) for acc, db, pk in qs.iterator()}

        # compile rxn<->compound relations
        lefts, rights = [], []
        for rxn_acc, rxndb, rxn_pk in rxn_qs.iterator():
            cpddb, left_accs, right_accs = compounds[rxn_acc]
            left = [(i, j) for i, j in (comp_acc2pk[k] for k in left_accs)]
            right = [(i, j) for i, j in (comp_acc2pk[k] for k in right_accs)]
            if not left and not right:
                continue

            # check db field aggreement
            comp_db = {i for i, _ in left + right}
            if len(comp_db) > 1:
                raise RuntimeError(
                    f'multiple compound DBs: {comp_db=} {rxn_acc=} {rxndb=} '
                    f'{left_accs=} {right_accs=} {left=} {right=}'
                )
            comp_db = comp_db.pop()
            if comp_db == rxndb or rxndb == ReactionEntry.DB_RHEA and comp_db == CompoundEntry.DB_CHEBI:  # noqa: E501
                # DBs match
                pass
            else:
                raise RuntimeError(
                    f'db field inconsistency: {comp_db=} {rxn_acc=} {rxndb=} '
                    f'{left_accs=} {right_accs=} {left=} {right=}'
                )
            lefts += [(rxn_pk, j) for _, j in left]
            rights += [(rxn_pk, j) for _, j in right]

        # save rxn<->compound relations
        for direc, rels in [('left', lefts), ('right', rights)]:
            through = ReactionEntry._meta.get_field(direc).remote_field.through
            print(f'Setting {len(rels)} {direc} reaction<->compound relations',
                  flush=True, end='')
            through_objs = [
                through(**{'reactionentry_id': i, 'compoundentry_id': j})
                for i, j in rels
            ]
            through.objects.bulk_create(through_objs)
            print(' [OK]')


class TaxName(Model):

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
    accession = AccessionField(prefix='TIGR')

    class Meta:
        verbose_name = 'TIGR'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class Uniprot(Model):
    accession = AccessionField(verbose_name='uniprot id')

    class Meta:
        verbose_name = 'Uniprot'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class UniRef100(Model):
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
    kegg = models.ManyToManyField(ReactionEntry, related_name='kegg_rxn')
    # 20 RHEA
    rhea = models.ManyToManyField(ReactionEntry, related_name='rhea_rxn')
    # 21 BIOCYC
    biocyc = models.ManyToManyField(ReactionEntry, related_name='bioc_rxn')
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
                print('[dry-run rollback]')

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
