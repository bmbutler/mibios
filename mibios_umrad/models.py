from collections import defaultdict
from itertools import groupby
from logging import getLogger
from pathlib import Path

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.transaction import atomic, set_rollback

from mibios import get_registry

from .fields import AccessionField
from . import manager
from .model_utils import (
    ch_opt, fk_req, VocabularyModel, delete_all_objects_quickly,
    LoadMixin, Model
)
from .utils import CSV_Spec


log = getLogger(__name__)


class CompoundRecord(Model):
    """ Reference DB's entry for chemical compound, reactant, or product """
    DB_BIOCYC = 'BC'
    DB_CHEBI = 'CH'
    DB_HMDB = 'HM'
    DB_PATHBANK = 'PB'
    DB_KEGG = 'KG'
    DB_PUBCHEM = 'PC'
    DB_CHOICES = (
        (DB_BIOCYC, 'Biocyc'),
        (DB_CHEBI, 'ChEBI'),
        (DB_HMDB, 'HMDB'),
        (DB_PATHBANK, 'PathBank'),
        (DB_KEGG, 'KEGG'),
        (DB_PUBCHEM, 'PubChem'),
    )

    accession = models.CharField(max_length=40, unique=True)
    source = models.CharField(max_length=2, choices=DB_CHOICES, db_index=True)
    formula = models.CharField(max_length=32, blank=True)
    charge = models.SmallIntegerField(blank=True, null=True)
    mass = models.CharField(max_length=16, blank=True)  # TODO: decimal??
    names = models.ManyToManyField('CompoundName')
    others = models.ManyToManyField('self', symmetrical=False)

    loader = manager.CompoundRecordLoader()

    def __str__(self):
        return f'{self.get_source_display()}:{self.accession}'

    def group(self):
        """ return QuerySet of synonym/related compound entry group """
        return self.compound.group.all()

    external_urls = {
        DB_BIOCYC: 'https://biocyc.org/compound?orgid=META&id={}',
        DB_CHEBI: 'https://www.ebi.ac.uk/chebi/searchId.do?chebiId={}',
        DB_HMDB: 'https://hmdb.ca/metabolites/{}',
        DB_PATHBANK: None,  # TODO
        DB_KEGG: 'https://www.kegg.jp/entry/{}',
        DB_PUBCHEM: (
            'https://pubchem.ncbi.nlm.nih.gov/compound/{}',
            lambda x: x.removeprefix('CID:')
        ),
    }

    def get_external_url(self):
        url_spec = self.external_urls[self.source]
        if not url_spec:
            return None
        elif isinstance(url_spec, str):
            # assume simple formatting string
            return url_spec.format(self.accession)
        else:
            # assumme a tuple (templ, func)
            return url_spec[0].format(url_spec[1](self.accession))


class CompoundName(VocabularyModel):
    max_length = 128
    abundance_accessor = 'compoundentry__abundance'


class FunctionName(VocabularyModel):
    max_length = 128
    abundance_accessor = 'funcrefdbentry__abundance'


class Location(VocabularyModel):
    class Meta(Model.Meta):
        verbose_name = 'subcellular location'


class Metal(VocabularyModel):
    pass


class ReactionEntryManager(manager.Manager):
    def create_from_m2m_input(self, values, source_model, src_field_name):
        if source_model is not UniRef100:
            raise NotImplementedError(
                'can only create instance on behalf of UniRef100'
            )
        if src_field_name == 'kegg_reactions':
            db = self.model.DB_KEGG
        elif src_field_name == 'rhea_reactions':
            db = self.model.DB_RHEA
        elif src_field_name == 'biocyc_reactions':
            db = self.model.DB_BIOCYC
        else:
            raise ValueError(f'unknown source field name: {src_field_name}')

        # create one unique reaction group per value
        try:
            last_pk = Reaction.objects.order_by('pk').latest('pk').pk
        except Reaction.DoesNotExist:
            last_pk = -1
        Reaction.objects.bulk_create(
            (Reaction() for _ in range(len(values))),
            batch_size=500,  # runs up at SQLITE_MAX_COMPOUND_SELECT
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
        (DB_BIOCYC, 'Biocyc'),
        (DB_KEGG, 'KEGG'),
        (DB_RHEA, 'RHEA'),
    )

    accession = AccessionField()
    db = models.CharField(max_length=1, choices=DB_CHOICES, db_index=True)
    bi_directional = models.BooleanField(blank=True, null=True)
    left = models.ManyToManyField(
        CompoundRecord, related_name='to_reaction',
    )
    right = models.ManyToManyField(
        CompoundRecord, related_name='from_reaction',
    )
    reaction = models.ForeignKey('Reaction', **fk_req)

    objects = ReactionEntryManager()

    def __str__(self):
        return self.accession


class Reaction(Model):
    """ distinct DB-independent reaction """

    # no fields here!

    loader_spec = CSV_Spec(
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

    loader = manager.ReactionLoader()

    class Meta:
        verbose_name = 'distinct reaction'


class FuncRefDBEntry(Model):
    DB_COG = 'cog'
    DB_EC = 'ec'
    DB_GO = 'go'
    DB_IPR = 'ipr'
    DB_PFAM = 'pfam'
    DB_TCDB = 'tcdb'
    DB_TIGR = 'tigr'
    DB_CHOICES = (
        (DB_COG, 'COG'),
        (DB_EC, 'EC'),
        (DB_GO, 'GO'),
        (DB_IPR, 'InterPro'),
        (DB_PFAM, 'Pfam'),
        (DB_TCDB, 'TCDB'),
        (DB_TIGR, 'TIGR'),
    )
    accession = AccessionField()
    db = models.CharField(max_length=4, choices=DB_CHOICES, db_index=True)
    names = models.ManyToManyField('FunctionName')

    loader = manager.FuncRefDBEntryLoader()

    class Meta(Model.Meta):
        verbose_name = 'Function Ref DB Entry'
        verbose_name_plural = 'Func Ref DB Entries'

    def __str__(self):
        return self.accession

    external_urls = {
        DB_COG: 'https://www.ncbi.nlm.nih.gov/research/cog/cog/{}/',
        DB_EC: '',
        DB_GO: 'http://amigo.geneontology.org/amigo/term/{}',
        DB_IPR: 'https://www.ebi.ac.uk/interpro/entry/InterPro/{}/',
        DB_PFAM: 'https://pfam.xfam.org/family/{}',
        DB_TCDB: '',
        DB_TIGR: '',
    }

    def get_external_url(self):
        url_spec = self.external_urls.get(self.db, None)
        if not url_spec:
            return None
        elif isinstance(url_spec, str):
            # assume simple formatting string
            return url_spec.format(self.accession)
        else:
            # assumme a tuple (templ, func)
            return url_spec[0].format(url_spec[1](self.accession))


class TaxID(Model):
    taxid = models.PositiveIntegerField(
        unique=True, verbose_name='NCBI taxid',
    )
    taxon = models.ForeignKey('Taxon', **fk_req)


class Taxon(Model):
    RANKS = (
        (0, 'root'),
        (1, 'domain'),
        (2, 'phylum'),
        (3, 'class'),
        (4, 'order'),
        (5, 'family'),
        (6, 'genus'),
        (7, 'species'),
        (8, 'strain'),
    )
    name = models.CharField(max_length=64)
    rank = models.PositiveSmallIntegerField(choices=RANKS)
    lineage = models.CharField(max_length=256)
    ancestors = models.ManyToManyField(
        'self',
        symmetrical=False,
        related_name='descendants',
    )

    loader = manager.TaxonLoader()

    class Meta(Model.Meta):
        unique_together = (
            ('rank', 'name'),
        )
        verbose_name_plural = 'taxa'

    def __str__(self):
        return f'{self.get_rank_display()} {self.name}'

    def get_parent(self):
        return self.ancestors.order_by('rank').last()

    def as_lineage(self):
        """ format taxon as lineage """
        parts = []
        ancestors = list(self.ancestors.order_by('rank'))
        for i, rank_name in self.RANKS[1:]:
            if ancestors:
                taxon = ancestors.pop(0)
            else:
                break
            if taxon.rank > i:
                try:
                    parts.append(f'UNCLASSIFIED_{parts[-1]}_{rank_name}')
                except IndexError:
                    print(f'BORK {self=} {parts=} {i=} {ancestors=} {taxon=}')
                    raise
            else:
                parts.append(taxon.name)
        return ';'.join(parts)

    @classmethod
    def get_search_field(cls):
        return cls._meta.get_field('name')

    @classmethod
    def get_lineage_rep_lookupper(cls):
        """
        helper to build lineage representation to object lookup dictionary

        Expensive to call, about 40s
        """
        objs = {obj.pk: obj for obj in cls.objects.iterator()}
        thru = cls._meta.get_field('ancestors').remote_field.through
        qs = thru.objects.values_list('from_taxon_id', 'to_taxon_id')
        qs = qs.order_by('from_taxon_id')
        lin2obj = {}
        for from_id, grp in groupby(qs.iterator(), key=lambda x: x[0]):
            rep = [(objs[i].rank, objs[i].name) for _, i in grp]
            rep.append((objs[from_id].rank, objs[from_id].name))
            lin2obj[tuple(rep)] = objs[from_id]
        return lin2obj

    @classmethod
    def get_parse_and_lookup_fun(cls):
        """
        Make and return a lineage string to Taxon instance mapper
        """
        lin2obj = cls.get_lineage_rep_lookupper()

        def str2instance(lineage):
            lineage = tuple(cls.parse_string(lineage))
            try:
                return lin2obj[lineage], None
            except KeyError:
                return None, lineage

        return str2instance

    @classmethod
    def parse_string(cls, lineage, sep=';'):
        """
        Parse a lineage string into list of (rank, name) pairs
        """
        specials = ('MICROBIOME',)
        lst = []
        check_species = False
        for rank, name in enumerate(lineage.split(sep), start=1):
            if rank < 7 and name.startswith('UNCLASSIFIED_'):
                # skipping these
                parts = name.split('_')
                if len(parts) == 2:
                    # unclass microbiome, others?
                    if parts[-1] not in specials:
                        raise RuntimeError(
                            f'can not handle: {rank=} {name=} in {lineage=}'
                        )
                    continue
                rank_name = parts[-1]
                if rank_name.casefold() != cls.RANKS[rank][1].casefold():
                    raise RuntimeError(
                        f'failed parsing: {rank=} {name=} rank mismatch'
                    )
                derivate = '_'.join(parts[1:-1])
                if derivate != lst[-1][1]:
                    raise RuntimeError(
                        f'failed parsing: uncl {rank=} {name=} {lst=} mismatch'
                    )
                continue

            if rank == 7 and name.endswith('_SP'):
                # test if _SP name is derived from higher rank
                for r, n in lst:
                    if name[:-3] == n:
                        skip = True
                        break
                else:
                    # not derived from higher rank
                    # keep this for now, but check again with strain
                    check_species = True
                    skip = False
                if skip:
                    continue

            if rank == 8 and check_species:
                # FIXME: not doing anything here for now
                pass

            lst.append((rank, name))
        return lst

    @classmethod
    def from_string(cls, lineage, obj_cache=None, sep=';'):
        """
        Get Taxon instance from a lineage string

        obj_cache: a dict: (rank, name)->Taxon

        ...
        """
        # formerly from_name_pks()
        *ancestry, (rank, name) = cls.parse_string(lineage, sep=sep)
        obj = cls(name=name, rank=rank, lineage=lineage)
        if obj_cache is None:
            q = None
            for r, n in ancestry:
                qi = models.Q(rank=r, name=n)
                q = qi if q is None else q | qi
            ancestry = Taxon.objects.filter(q)
        else:
            ancestry = [obj_cache[(r, n)] for r, n in ancestry]

        return obj


class Uniprot(Model):
    accession = AccessionField(verbose_name='uniprot id')

    class Meta(Model.Meta):
        verbose_name = 'Uniprot'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    def get_external_url(self):
        return f'https://www.uniprot.org/uniprot/{self.accession}'


class UniRef100(LoadMixin, Model):
    """
    Model for UniRef100 clusters
    """
    # The field comments below are based on the columns in
    # UNIREF100_INFO_DEC_2021.txt in order.

    #  1 UNIREF100
    accession = AccessionField(prefix='UNIREF100_')
    #  2 NAME
    function_names = models.ManyToManyField(FunctionName)
    #  3 LENGTH
    length = models.PositiveIntegerField(blank=True, null=True)
    #  4 UNIPROT_IDS
    uniprot = models.ManyToManyField(Uniprot)
    #  5 UNIREF90
    uniref90 = AccessionField(prefix='UNIREF90_', unique=False)
    #  6 TAXON_IDS
    taxids = models.ManyToManyField(TaxID, related_name='classified_uniref100')
    #  7 LINEAGE
    lineage = models.ForeignKey(Taxon, **fk_req)
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
    subcellular_locations = models.ManyToManyField(Location)
    # 14-19 COG PFAM TIGR GO IPR EC
    function_refs = models.ManyToManyField(FuncRefDBEntry)
    # 20-22 KEGG RHEA BIOCYC
    kegg_reactions = models.ManyToManyField(
        ReactionEntry,
        related_name='uniref_kegg',
    )
    rhea_reactions = models.ManyToManyField(
        ReactionEntry,
        related_name='uniref_rhea',
    )
    biocyc_reactions = models.ManyToManyField(
        ReactionEntry,
        related_name='uniref_biocyc',
    )
    # 23 REACTANTS
    # 24 PRODUCTS
    # 25 TRANS_CPD
    trans_compounds = models.ManyToManyField(
        CompoundRecord,
        related_name='uniref_trans',
    )

    class Meta:
        verbose_name = 'UniRef100'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    import_file_spec = (
        ('UNIREF100', 'accession'),
        ('NAME', 'function_names'),
        ('LENGTH', 'length'),
        ('UNIPROT_IDS', 'uniprot'),
        ('UNIREF90', 'uniref90'),
        ('TAXON_IDS', 'taxids'),
        ('LINEAGE', 'lineage'),
        ('SIGALPEP', 'signal_peptide'),
        ('TMS', 'tms'),
        ('DNA', 'dna_binding'),
        ('METAL', 'metal_binding'),
        ('TCDB', 'tcdb'),
        ('LOCATION', 'subcellular_locations'),
        ('COG', FuncRefDBEntry.DB_COG),
        ('PFAM', FuncRefDBEntry.DB_PFAM),
        ('TIGR', FuncRefDBEntry.DB_TIGR),
        ('GO', FuncRefDBEntry.DB_GO),
        ('IPR', FuncRefDBEntry.DB_IPR),
        ('EC', FuncRefDBEntry.DB_EC),
        ('KEGG', 'kegg_reactions',),
        ('RHEA', 'rhea_reactions'),
        ('BIOCYC', 'biocyc_reactions'),
        ('REACTANTS', None),
        ('PRODUCTS', None),
        ('TRANS_CPD', 'trans_compounds'),
    )

    @classmethod
    def get_file(cls):
        if 'UNIREF100_INFO_PATH' in dir(settings):
            return settings.UNIREF100_INFO_PATH
        else:
            # FIXME
            return (Path(settings.UMRAD_ROOT) / 'UNIPROT'
                    / 'UNIREF100_INFO_DEC_2021.txt')

    @classmethod
    @atomic
    def load(cls, max_rows=None, start=0, dry_run=False):
        # get data and split m2m fields
        refdb_keys = [i for i, _ in FuncRefDBEntry.DB_CHOICES]
        rxndb_keys = [i for i, _ in ReactionEntry.DB_CHOICES]
        field_names = [i.name for i in cls._meta.get_fields()]

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

        # get lookups for FK PKs
        print('Retrieving lineage data... ', end='', flush=True)
        get_taxon = Taxon.get_parse_and_lookup_fun()
        print('[OK]')

        objs = []
        m2m_data = {}
        xref_data = defaultdict(list)  # maps a ref DB references to UniRef100s
        new_taxa = defaultdict(list)

        data = super().load(max_rows=max_rows, start=start, parse_only=True)
        for row in data:
            obj = cls()
            m2m = {}
            xrefs = []  # a list of pairs: (db, xref list)
            for key, value in row.items():

                if value == '':
                    continue

                if key in field_names:
                    if key in m2mcols:
                        # regular m2m fields
                        m2m[key] = value
                    elif key == 'lineage':
                        taxon, new_rep = get_taxon(value)
                        if taxon is None:
                            # taxon not found in DB
                            # save with index so we find the obj later
                            new_taxa[new_rep].append(len(objs))
                        else:
                            obj.lineage = taxon
                    else:
                        # regular field (length, dna_binding, ...)
                        setattr(obj, key, value)
                elif key in m2mcols and key in refdb_keys:
                    # ref DB references
                    xrefs.append((key, cls._split_m2m_input(value)))
                else:
                    raise RuntimeError(
                        f'a bug, other cases were supposed to be'
                        f'exhaustive: {key=} {field_names=} {m2mcols=} '
                        f'{refdb_keys=}'
                    )

            acc = obj.get_accession_single()
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

        if new_taxa:
            print(f'NEW TAXA: {len(new_taxa)=}', str(new_taxa)[:500])
        if False and new_taxa:
            # FIXME and what does it mean to have stuff here but not in Taxon?
            # create+save+reload new taxa, then set missing PKs in unirefs
            try:
                maxpk = Taxon.objects.latest('pk').pk
            except Taxon.DoesNotExist:
                maxpk = 0
            Taxon.objects.bulk_create(
                (Taxon.from_string(i) for i in new_taxa.keys())
            )
            for i in Taxon.objects.filter(pk__gt=maxpk):
                for j in new_taxa[i.get_name_pks()]:
                    objs[j].lineage_id = i.pk  # set lineage PK to UniRef obj
            del maxpk

        m2m_fields = list(m2m.keys())
        del m2m

        cls.objects.bulk_create(objs)

        # get accession -> pk map
        acc2pk = dict(
            cls.objects
            .values_list(cls.get_accession_lookup_single(), 'pk')
            .iterator()
        )

        # replace accession with pk in m2m data keys
        m2m_data = {acc2pk[i]: data for i, data in m2m_data.items()}

        # collecting all m2m entries
        for i in m2m_fields:
            cls._update_m2m(i, m2m_data)
        del m2m_data

        # store new xref entries
        existing_xrefs = set(
            FuncRefDBEntry.objects
            .values_list('accession', flat=True)
            .iterator()
        )
        xref_objs = (FuncRefDBEntry(accession=i, db=db)
                     for (i, db) in xref_data.keys()
                     if i not in existing_xrefs)
        FuncRefDBEntry.objects.bulk_create(xref_objs)
        del existing_xrefs

        # get PKs for xref objects
        xref2pk = dict(
            FuncRefDBEntry.objects.values_list('accession', 'pk').iterator()
        )

        # store UniRef100 <-> FuncRefDBEntry relations
        rels = (
            (acc2pk[i], xref2pk[xref])
            for (xref, _), accs in xref_data.items()
            for i in accs
        )
        through = cls._meta.get_field('function_refs').remote_field.through
        through_objs = (
            through(uniref100_id=i, funcrefdbentry_id=j)
            for i, j in rels
        )
        through_objs = list(through_objs)
        manager.Manager.bulk_create_wrapper(through.objects.bulk_create)(through_objs)  # noqa:E501

        set_rollback(dry_run)

    def get_external_url(self):
        return f'https://www.uniprot.org/uniref/{self.accession}'


# development stuff
def delete_all_uniref100_etc():
    r = get_registry()
    for i in r.apps['mibios_umrad'].get_models():
        if i._meta.model_name.startswith('tax'):
            continue
        print(f'Deleting: {i} ', end='', flush=True)
        delete_all_objects_quickly(i)
        print('[done]')


def load_umrad():
    """ load all of UMRAD from scratch, assuming an empty DB """
    CompoundRecord.loader.load()
    Reaction.loader.load()
    Taxon.loader.load()
    FuncRefDBEntry.loader.load()
    UniRef100.load()
