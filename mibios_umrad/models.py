from collections import OrderedDict, defaultdict
from itertools import groupby, zip_longest
from logging import getLogger
from operator import itemgetter
from pathlib import Path

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.transaction import atomic
from django.utils.functional import cached_property

from mibios import get_registry


from .fields import AccessionField
from .model_utils import (
    ch_opt, fk_opt, fk_req, VocabularyModel, delete_all_objects_quickly,
    Manager, Model,
)
from .utils import ProgressPrinter, DryRunRollback


log = getLogger(__name__)


class CompoundEntryManager(Manager):
    def create_from_m2m_input(self, values, source_model, source_field_name):
        if source_model is UniRef100 and source_field_name == 'trans_compound':
            pass
        else:
            raise NotImplementedError(
                'is only implemented for field UniRef100.trans_compound'
            )

        # create one unique reaction group per value
        try:
            last_pk = Compound.objects.order_by('pk').latest('pk').pk
        except Compound.DoesNotExist:
            last_pk = -1
        Compound.objects.bulk_create((Compound() for _ in range(len(values))))
        cpd_pks = Compound.objects.filter(pk__gt=last_pk)\
                          .values_list('pk', flat=True)
        if len(values) != len(cpd_pks):
            # just checking
            raise RuntimeError('a bug making right number of Compound objects')

        model = self.model
        db = CompoundEntry.DB_CHEBI
        objs = (model(accession=i, db=db, compound_id=j)
                for i, j in zip(values, cpd_pks))
        return self.bulk_create(objs)


class CompoundEntry(Model):
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

    objects = CompoundEntryManager()

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


class FunctionName(VocabularyModel):
    max_length = 128


class Location(VocabularyModel):
    pass

    class Meta:
        verbose_name = 'subcellular location'


class Metal(VocabularyModel):
    pass


class ReactionEntryManager(Manager):
    def create_from_m2m_input(self, values, source_model, source_field_name):
        if source_model is not UniRef100:
            raise NotImplementedError(
                'can only create instance on behalf of UniRef100'
            )
        if source_field_name == 'kegg_reaction':
            db = self.model.DB_KEGG
        elif source_field_name == 'rhea_reaction':
            db = self.model.DB_RHEA
        elif source_field_name == 'biocyc_reaction':
            db = self.model.DB_BIOCYC
        else:
            raise ValueError(f'unknown source field name: {source_field_name}')

        # create one unique reaction group per value
        try:
            last_pk = Reaction.objects.order_by('pk').latest('pk').pk
        except Reaction.DoesNotExist:
            last_pk = -1
        Reaction.objects.bulk_create(
            (Reaction() for _ in range(len(values))),
            # FIXME: keep getting "too many terms in compound SELECT" for >500
            batch_size=500,
        )
        reaction_pks = Reaction.objects.filter(pk__gt=last_pk)\
                               .values_list('pk', flat=True)
        if len(values) != len(reaction_pks):
            # just checking
            raise RuntimeError('a bug making right number of Reaction objects')

        model = self.model
        objs = (model(accession=i, db=db, reaction_id=j)
                for i, j in zip(values, reaction_pks))
        return self.bulk_create(objs)


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
    bi_directional = models.BooleanField(blank=True, null=True)
    left = models.ManyToManyField(
        CompoundEntry, related_name='to_reaction',
    )
    right = models.ManyToManyField(
        CompoundEntry, related_name='from_reaction',
    )
    reaction = models.ForeignKey('Reaction', **fk_req)

    objects = ReactionEntryManager()

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
        m2mcols = [i for _, i in cls.import_file_spec if i not in ['accession', 'dir']]  # noqa:E501
        for row in super().load(max_rows=max_rows, parse_only=True):
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


class RefDBEntry(Model):
    DB_COG = 'cog'
    DB_EC = 'ec'
    DB_GO = 'go'
    DB_IPR = 'ipr'
    DB_PFAM = 'pfam'
    DB_TCDB = 'tcdb'
    DB_TIGR = 'tigr'
    DB_CHOICES = (
        (DB_COG, DB_COG),
        (DB_EC, DB_EC),
        (DB_GO, DB_GO),
        (DB_IPR, DB_IPR),
        (DB_PFAM, DB_PFAM),
        (DB_TCDB, DB_TCDB),
        (DB_TIGR, DB_TIGR),
    )
    accession = AccessionField()
    db = models.CharField(max_length=4, db_index=True)

    class Meta:
        verbose_name = 'Ref DB Entry'
        verbose_name_plural = 'Ref DB Entries'

    def __str__(self):
        return self.accession


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


class Uniprot(Model):
    accession = AccessionField(verbose_name='uniprot id')

    class Meta:
        verbose_name = 'Uniprot'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession


class UniRef100(Model):
    """
    Model for UniRef100 clusters
    """
    # The field comments below are based on the columns in
    # UNIREF100_INFO_DEC_2021.txt in order.

    #  1 UNIREF100
    accession = AccessionField(prefix='UNIREF100_')
    #  2 NAME
    function_name = models.ManyToManyField(FunctionName)
    #  3 LENGTH
    length = models.PositiveIntegerField(blank=True, null=True)
    #  4 UNIPROT_IDS
    uniprot = models.ManyToManyField(Uniprot)
    #  5 UNIREF90
    uniref90 = AccessionField(prefix='UNIREF90_', unique=False)
    #  6 TAXON_IDS
    taxon = models.ManyToManyField(Taxon)
    #  7 LINEAGE (method)
    #  8 SIGALPEP
    signal_peptide = models.CharField(max_length=32, **ch_opt)
    #  9 TMS
    tms = models.CharField(max_length=128, **ch_opt)
    # 10 DNA
    dna_binding = models.CharField(max_length=128, **ch_opt)
    # 11 METAL
    metal_binding = models.ManyToManyField(Metal)
    # 12 TCDB
    tcdb = models.CharField(max_length=128, **ch_opt)  # TODO: what is this?
    # 13 LOCATION
    subcellular_location = models.ManyToManyField(Location)
    # 14-19 COG PFAM TIGR GO IPR EC
    function_ref = models.ManyToManyField(RefDBEntry)
    # 20-22 KEGG RHEA BIOCYC
    kegg_reaction = models.ManyToManyField(
        ReactionEntry,
        related_name='uniref_kegg',
    )
    rhea_reaction = models.ManyToManyField(
        ReactionEntry,
        related_name='uniref_rhea',
    )
    biocyc_reaction = models.ManyToManyField(
        ReactionEntry,
        related_name='uniref_biocyc',
    )
    # 23 REACTANTS
    # 24 PRODUCTS
    # 25 TRANS_CPD
    trans_compound = models.ManyToManyField(
        CompoundEntry,
        related_name='uniref_trans',
    )

    class Meta:
        verbose_name = 'UniRef100'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    import_file_spec = (
        ('UNIREF100', 'accession'),
        ('NAME', 'function_name'),
        ('LENGTH', 'length'),
        ('UNIPROT_IDS', 'uniprot'),
        ('UNIREF90', 'uniref90'),
        ('TAXON_IDS', None),  # 'taxon'),
        ('LINEAGE', None),
        ('SIGALPEP', 'signal_peptide'),
        ('TMS', 'tms'),
        ('DNA', 'dna_binding'),
        ('METAL', 'metal_binding'),
        ('TCDB', 'tcdb'),
        ('LOCATION', 'subcellular_location'),
        ('COG',  RefDBEntry.DB_COG),
        ('PFAM', RefDBEntry.DB_PFAM),
        ('TIGR', RefDBEntry.DB_TIGR),
        ('GO', RefDBEntry.DB_GO),
        ('IPR', RefDBEntry.DB_IPR),
        ('EC', RefDBEntry.DB_EC),
        ('KEGG', 'kegg_reaction',),
        ('RHEA', 'rhea_reaction'),
        ('BIOCYC', 'biocyc_reaction'),
        ('REACTANTS', None),
        ('PRODUCTS', None),
        ('TRANS_CPD', 'trans_compound'),
    )

    def lineage(self):
        return ';'.join(Taxon.lcs_lineage(self.taxon_set.all()))

    @classmethod
    def get_file(cls):
        return (Path(settings.UMRAD_ROOT) / 'UNIPROT'
                / 'UNIREF100_INFO_DEC_2021.txt')

    @classmethod
    @atomic
    def load(cls, max_rows=None, dry_run=True):
        # get data and split m2m fields
        refdb_keys = [i for _, i in RefDBEntry.DB_CHOICES]
        rxndb_keys = [i for _, i in ReactionEntry.DB_CHOICES]
        field_names = [i.name for i in cls._meta.get_fields()]
        accession_field_name = cls.get_accession_field().name

        m2mcols = []
        for _, i in cls.import_file_spec:
            try:
                field = cls._meta.get_field(i)
            except FieldDoesNotExist:
                if i in refdb_keys or i in rxndb_keys:
                    m2mcols.append(i)
            else:
                if field.many_to_many:
                    m2mcols.append(i)
        del field

        objs = []
        m2m_data = {}
        xref_data = defaultdict(list)  # maps a ref DB references to UniRef100s

        for row in super().load(max_rows=max_rows, parse_only=True):
            obj = cls()
            m2m = {}
            xrefs = []  # a list of pairs: (db, xref list)
            for key, value in row.items():

                if key in field_names:
                    # regular fields
                    if key in m2mcols:
                        m2m[key] = value
                    elif value != '':
                        setattr(obj, key, value)
                elif key in m2mcols and key in refdb_keys:
                    # ref DB references
                    xrefs.append((key, cls._split_m2m_input(value)))
                else:
                    raise RuntimeError(
                        f'a bug, other cases were supposed to be'
                        f'exhaustive: {key=}'
                    )

            acc = getattr(obj, accession_field_name)
            if acc in m2m_data:
                # duplicate row !!?!??
                print(f'WARNING: skipping row with duplicate UniRef100 '
                      f'accession: {acc}')
                continue

            m2m_data[acc] = m2m
            objs.append(obj)
            for dbkey, values in xrefs:
                for i in values:
                    xref_data[(i, dbkey)].append(acc)

        del row, key, value, values, xrefs, acc, dbkey

        m2m_fields = list(m2m.keys())
        del m2m

        cls.objects.bulk_create(objs)

        # get accession -> pk map
        acc2pk = dict(
            cls.objects.values_list(accession_field_name, 'pk').iterator()
        )

        # replace accession with pk in m2m data keys
        m2m_data = {acc2pk[i]: data for i, data in m2m_data.items()}

        # collecting all m2m entries
        for i in m2m_fields:
            cls._update_m2m(i, m2m_data)
        del m2m_data

        # store xref entries (assume none exist yet)
        xref_objs = (RefDBEntry(accession=i, db=db)
                     for (i, db) in xref_data.keys())
        RefDBEntry.objects.bulk_create(xref_objs)

        # get PKs for xref objects
        xref2pk = dict(
            RefDBEntry.objects.values_list('accession', 'pk').iterator()
        )

        # store UniRef100 <-> RefDBEntry relations
        rels = (
            (acc2pk[i], xref2pk[xref])
            for (xref, _), accs in xref_data.items()
            for i in accs
        )
        through = cls._meta.get_field('function_ref').remote_field.through
        through_objs = (
            through(uniref100_id=i, refdbentry_id=j)
            for i, j in rels
        )
        through_objs = list(through_objs)
        through.objects.bulk_create(through_objs)
        print(f'created {len(through_objs)} UniRef100 vs xref entry relations')

        if dry_run:
            raise DryRunRollback


# development stuff
def delete_all_uniref100_etc():
    r = get_registry()
    for i in r.apps['mibios_umrad'].get_models():
        if i._meta.model_name.startswith('tax'):
            continue
        print(f'Deleting: {i} ', end='', flush=True)
        delete_all_objects_quickly(i)
        print('[done]')
