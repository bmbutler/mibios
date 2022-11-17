from itertools import groupby
from logging import getLogger

from django.db import models

from mibios import get_registry

from .fields import AccessionField
from . import manager
from .model_utils import (
    ch_opt, fk_opt, fk_req, VocabularyModel, delete_all_objects_quickly,
    Model
)


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


class ReactionCompound(models.Model):
    """
    intermediate model for reaction -> l/r-compound m2m relations

    This is an ordinary django.db.models.Model
    """
    DB_BIOCYC = 'BC'
    DB_KEGG = 'KG'
    DB_RHEA = 'CH'
    SRC_CHOICES = (
        (DB_BIOCYC, 'Biocyc'),
        (DB_KEGG, 'KEGG'),
        (DB_RHEA, 'RHEA'),
    )
    SIDE_CHOICES = ((True, 'left'), (False, 'right'))
    LOCATION_CHOICES = ((True, 'INSIDE'), (False, 'OUTSIDE'))
    TRANSPORT_CHOICES = (
        ('BI', 'BIPORT'),
        ('EX', 'EXPORT'),
        ('IM', 'IMPORT'),
        ('NO', 'NOPORT'),
    )

    # source and target field names correspond to what an automatic through
    # model would have
    reactionrecord = models.ForeignKey('ReactionRecord', **fk_req)
    compoundrecord = models.ForeignKey('CompoundRecord', **fk_req)
    source = models.CharField(max_length=2, choices=SRC_CHOICES, db_index=True)
    side = models.BooleanField(choices=SIDE_CHOICES)
    location = models.BooleanField(choices=LOCATION_CHOICES)
    transport = models.CharField(max_length=2, choices=TRANSPORT_CHOICES)

    class Meta:
        unique_together = (
            ('reactionrecord', 'compoundrecord', 'source', 'side'),
        )

    def __str__(self):
        return (
            f'{self.reactionrecord.accession}<>{self.compoundrecord.accession}'
            f' {self.source} {self.get_side_display()}'
        )


class ReactionRecord(Model):
    SRC_CHOICES = ReactionCompound.SRC_CHOICES
    DIRECT_CHOICES = (
        (True, 'BOTH'),
        (False, 'LTR'),
    )
    accession = AccessionField(max_length=96)
    source = models.CharField(max_length=2, choices=SRC_CHOICES, db_index=True)
    direction = models.BooleanField(
        choices=DIRECT_CHOICES,
        blank=True,
        null=True,
    )
    others = models.ManyToManyField('self', symmetrical=False)
    compound = models.ManyToManyField(
        CompoundRecord,
        through=ReactionCompound,
    )
    uniprot = models.ManyToManyField('Uniprot')
    ec = models.ForeignKey('FuncRefDBEntry', **fk_opt)

    loader = manager.ReactionRecordLoader()

    def __str__(self):
        return self.accession


class FuncRefDBEntry(Model):
    DB_COG = 'cg'
    DB_EC = 'ec'
    DB_GO = 'go'
    DB_IPR = 'ip'
    DB_PFAM = 'pf'
    DB_TCDB = 'tc'
    DB_TIGR = 'ti'
    DB_CHOICES = (
        # by order of UNIREF input file columns
        (DB_TCDB, 'TCDB'),
        (DB_COG, 'COG'),
        (DB_PFAM, 'Pfam'),
        (DB_TIGR, 'TIGR'),
        (DB_GO, 'GO'),
        (DB_IPR, 'InterPro'),
        (DB_EC, 'EC'),
    )
    accession = AccessionField()
    db = models.CharField(max_length=2, choices=DB_CHOICES, db_index=True)
    names = models.ManyToManyField('FunctionName')

    name_loader = manager.FuncRefDBEntryLoader()

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
        DB_TIGR: '',
        DB_TCDB: '',
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
    taxid = models.PositiveBigIntegerField(
        unique=True, verbose_name='NCBI taxid',
    )
    taxon = models.ForeignKey('Taxon', **fk_req)

    # disable auto-creation as m2m target
    loader = None

    def __str__(self):
        return str(self.taxid)

    def get_external_url(self):
        return (f'https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id='
                f'{self.taxid}')


class Taxon(Model):
    """
    A taxonomic tree

    Implementation of a taxonomic tree with 9 ranks.  Each node is a (rank,
    name) pair.  Except for the root, the ancestor relation is the transitive
    closure of the parent/child links.  Not all ranks have to be present in a
    lineage.  For efficiency, relations to the root are implied, that is, the
    root is not related to any other node.
    """
    RANKS = (
        (0, 'root'),  # implied only, no root object saved in DB
        (1, 'domain'),
        (2, 'phylum'),
        (3, 'class'),
        (4, 'order'),
        (5, 'family'),
        (6, 'genus'),
        (7, 'species'),
        (8, 'strain'),
    )
    name = models.CharField(max_length=256)
    rank = models.PositiveSmallIntegerField(choices=RANKS)
    lineage = models.CharField(max_length=512)
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
        """
        Get the nearest ancestor, the parent.

        Returns None for the root
        """
        parent = self.ancestors.order_by('rank').last()
        if parent is None:
            # no ancestors
            if self.rank == 0:
                # we're root, no parent
                return None
            else:
                # root is parent
                return Taxon.objects.get(rank=0)
        return parent

    def as_lineage(self, to_species=False):
        """
        format taxon as lineage

        Start with kingdom ranks, fill in gaps with placeholder names.

        to_species: for higher-level taxa, if needed, auto-fill placeholder
                    names until species
        """
        # last: remembers the stem to be used for unclassified parts
        last = None
        nodes = list(self.ancestors.order_by('rank')) + [self]
        parts = []
        for i, rank_name in self.RANKS[1:]:
            if nodes:
                taxon = nodes[0]
                if taxon.rank > i:
                    if i == 7:
                        parts.append(f'{last}_SP')
                    else:
                        parts.append(
                            f'UNCLASSIFIED_{last}_{rank_name.upper()}'
                        )
                else:
                    parts.append(taxon.name)
                    last = taxon.name
                    nodes.pop(0)

            elif to_species and i < 8:
                if i < 7:
                    # auto-fill part before species
                    parts.append(f'UNCLASSIFIED_{last}_{rank_name.upper()}')
                elif i == 7:
                    # auto-fill species and be done
                    parts.append(f'{last}_SP')
                    break
                else:
                    raise RuntimeError('a logic bug')
            else:
                # all done
                break

        return ';'.join(parts)

    @classmethod
    def get_search_field(cls):
        return cls._meta.get_field('name')

    @classmethod
    def get_lineage_rep_lookupper(cls):
        """
        helper to build lineage-representation-to-object lookup dictionary

        Expensive to call, about 40s
        """
        root = cls.objects.get(rank=0)
        objs = {obj.pk: obj for obj in cls.objects.iterator()}
        lin2obj = {}

        # empty lineage maps to root
        lin2obj[()] = cls.objects.get(rank=0)

        # orphans / phylum-level nodes below root (no ancestors, remember:
        # unrelated root)
        for i in cls.objects.filter(ancestors=None, rank__gt=0):
            lin2obj[((i.rank, i.name), )] = i

        # UNKNOWN_ special cases: suggesting length-2 lineages: [<root>;<org>]
        # NOTE: these have not shown up in UNIREF_INFO nor _contigs_ files yet
        for i in root.descendants.all():
            lin = ((root.rank, root.name), (i.rank, i.name))
            lin2obj[lin] = i

        # regular inner nodes (not UNKNOWN_[...], which link to root)
        # for each inner node, collect all ancestors -> lineage
        thru = cls._meta.get_field('ancestors').remote_field.through
        qs = thru.objects.exclude(to_taxon=root)
        qs = qs.values_list('from_taxon_id', 'to_taxon_id')
        qs = qs.order_by('from_taxon_id')
        for from_id, grp in groupby(qs.iterator(), key=lambda x: x[0]):
            lin = [(objs[i].rank, objs[i].name) for _, i in grp]
            lin.append((objs[from_id].rank, objs[from_id].name))
            lin2obj[tuple(lin)] = objs[from_id]

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

        For an empty string an empty list is returned.
        """
        if not sep:
            raise ValueError('separator must not be empty')

        if not lineage:
            # avoid ''.split(sep) -> ['']
            return []

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


class UniRef100(Model):
    """
    Model for UniRef100 clusters
    """
    # The field comments below are based on the columns in
    # OUT_UNIREF.txt in order.

    #  1 UR100
    accession = AccessionField(prefix='UNIREF100_')
    #  2 UR90
    uniref90 = AccessionField(prefix='UNIREF90_', unique=False)
    #  3 Name
    function_names = models.ManyToManyField(FunctionName)
    #  4 Length
    length = models.PositiveIntegerField(blank=True, null=True)
    #  5 SigPep
    signal_peptide = models.CharField(max_length=32, **ch_opt)
    #  6 TMS
    tms = models.CharField(max_length=128, **ch_opt)
    #  7 DNA
    dna_binding = models.CharField(max_length=128, **ch_opt)
    #  8 TaxonId
    taxids = models.ManyToManyField(TaxID, related_name='classified_uniref100')
    #  9 Metal
    metal_binding = models.ManyToManyField(Metal)
    # 10 Loc
    subcellular_locations = models.ManyToManyField(Location)
    # 11-17 TCDB COG Pfam Tigr Gene_Ont InterPro ECs
    function_refs = models.ManyToManyField(FuncRefDBEntry)
    # 18-20 kegg rhea biocyc
    reactions = models.ManyToManyField(ReactionRecord)

    loader = manager.UniRef100Loader()

    class Meta:
        verbose_name = 'UniRef100'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.accession

    def get_external_url(self):
        # ensure correct casing for UniRef100 prefix
        _, _, accession = self.accession.partition('_')
        return f'https://www.uniprot.org/uniref/UniRef100_{accession}'


# development stuff
def delete_all_uniref100_etc():
    r = get_registry()
    for i in r.apps['mibios.umrad'].get_models():
        if i._meta.model_name.startswith('tax'):
            continue
        print(f'Deleting: {i} ', end='', flush=True)
        delete_all_objects_quickly(i)
        print('[done]')


def load_umrad():
    """ load all of UMRAD from scratch, assuming an empty DB """
    Taxon.loader.load()
    CompoundRecord.loader.load(skip_on_error=True)
    ReactionRecord.loader.load(skip_on_error=True)
    UniRef100.loader.load(skip_on_error=True)
    FuncRefDBEntry.name_loader.load()
