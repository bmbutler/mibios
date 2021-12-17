from itertools import zip_longest
from pathlib import Path

from django.db import models

from mibios.models import Model
from mibios_umrad.model_utils import ch_opt, fk_opt, fk_req, opt
from . import get_data_source_path


class NCBITaxModel(Model):
    """ the mibios model without history """
    history = None

    class Meta(Model.Meta):
        abstract = True

    # name of downloaded dump file
    source_file = None

    @classmethod
    def load(cls):
        """
        Load data from NCBI taxdump files.

        This general load() method encapsulates most of the logic that governs
        the taxonomy data and should work with all relevant dmp files.
        """
        fields = cls.get_fields(skip_auto=True, with_m2m=True).fields
        # need to skip m2m-relations (else we get both sides of a m2m):
        fields = [i for i in fields
                  if not (i.many_to_many and not isinstance(i, models.Field))]
        here_field = cls.get_here_field()  # cf. to_field

        # fks: store of all possible value->key pairs that may occur in the
        # source data for the FK column.  We assume here that that data is
        # available, that is, the table which the FK points to, must have been
        # loaded already.  And that means no such circular relationship (i.e.
        # FK on self and m2m) is supported.  Those have to be handled using the
        # "rest data" that load() returns.
        fks = {}
        for i in [i for i in fields if i.many_to_one]:
            if i.related_model is cls:
                continue
            fks[i] = (
                i.related_model.get_here_field(),  # to run to_python on
                i.related_model.get_pk_for_keys(),  # this is the val->pk map
            )

        objs = []
        unprocessed = []
        skipped = 0
        for i, row in cls.data_source_rows():
            if len(fields) != len(row):
                raise RuntimeError(
                    f'{len(fields)=} != {len(row)=} at\n{i=} {fields=} {row=}'
                )

            skip_row = False
            kw = {}
            rest = {}
            for f, value in zip(fields, row):
                attr_name = f.name

                if value is None:
                    continue
                if f.many_to_many:
                    rest[f] = value
                    continue
                elif f.many_to_one and f.related_model is cls:
                    # FK to self
                    rest[f] = value
                    continue
                elif f.many_to_one:
                    # is normal FK, so assign pk
                    attr_name = f.name + '_id'
                    try:
                        value = fks[f][1][fks[f][0].to_python(value)]
                    except KeyError as e:
                        # happens with TypeMaterial
                        skipped += 1
                        if skipped <= 6:
                            print(
                                f'WARNING: (KeyError) {e} -- not a valid '
                                f'reference to {f.related_model} -- will skip '
                                f'line {i} in {cls.source_file}\nstate at '
                                f'point of error: {cls=} {f=} {i=} {value=} '
                                f'{fks[f][0]=}'
                            )
                        if skipped == 6:
                            print('(no further such warnings will be printed)')
                        skip_row = True
                        break

                if f.choices:
                    value = cls.lookup_choice(f, value)

                kw[attr_name] = value

            if skip_row:
                continue

            objs.append(cls(**kw))
            if here_field is not None and rest:
                # assumes here_field is no FK:
                unprocessed.append((kw[here_field.name], rest))

        if skipped:
            print(f'WARNING: altogether skipped {skipped} lines due to invalid'
                  f' references (see above for messages)')

        cls.objects.bulk_create(objs)
        return unprocessed

    @classmethod
    def data_source_rows(cls, path=None):
        """
        generate rows over ncbi taxonomy db files

        Yields tuples of line number and row
        """
        if cls.source_file is None:
            raise RuntimeError(
                f'no data source file set up: model {cls} can not import data'
            )
        if path is None:
            path = get_data_source_path() / cls.source_file

        with Path(path).open() as f:
            for i, line in enumerate(f, start=1):
                line = line.rstrip('\n')
                if line.endswith('\t|'):
                    # most lines, strip "\t|"
                    line = line[:-2]
                row = line.split('\t|\t')
                row = [i.strip() for i in row]
                row = [None if i == '' else i for i in row]
                yield i, row

    @classmethod
    def get_pk_for_keys(cls):
        """
        Helper to assign FKs on import

        Returns a dict (ncbi key)->pk
        """
        # FIXME: Put this method into the manager later
        key = cls.get_here_field()
        if key is None:
            raise RuntimeError(f'no key field for {cls}')
        return dict(cls.objects.values_list(key.name, 'pk').iterator())

    @classmethod
    def get_here_field(cls):
        """
        Get the unique (non-pk) field that NCBI uses for internal references

        Think of this as the reverse of the FK's to_field. We don't set
        to_field in any model but are used in the NCBI taxonomy and we use it
        to keep track of rows before the pk/id is known.

        Returns None if we don't have such a field (TaxName only?).
        This is usually the first non auto-pk unique field.
        """
        for i in cls.get_fields(skip_auto=True).fields:
            if i.unique:
                return i
        return None

    @classmethod
    def lookup_choice(cls, field, value):
        """ Get a field's choices db value from give human-readable value """
        if not hasattr(cls, '_field_choice_map'):
            cls._field_choice_map = {}

        if isinstance(field, str):
            field = cls._meta.get_field(field)

        if not field.choices:
            raise ValueError(f'field {field} choices must not be empty')

        if field not in cls._field_choice_map:
            cls._field_choice_map[field] = \
                dict(((b, a) for a, b in field.choices))

        return cls._field_choice_map[field][value]


class Citation(NCBITaxModel):
    cit_id = models.PositiveIntegerField(
        unique=True,
        help_text='the unique id of citation',
    )
    cit_key = models.CharField(
        max_length=512,
        help_text='citation key',
    )
    medline_id = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text='unique id in MedLine database',
    )
    pubmed_id = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text='unique id in PubMed database',
    )
    url = models.CharField(
        max_length=512,
        help_text='URL associated with citation',
    )
    text = models.CharField(
        max_length=1024,
        help_text='any text (usually article name and authors)'
        """
            The following characters are escaped in this text by a backslash
            newline (appear as "\n"),
            tab character ("\t"),
            double quotes ('\"'),
            backslash character ("\\").
        """
    )
    node = models.ManyToManyField('TaxNode')

    source_file = 'citations.dmp'

    # 0 means blank for these:
    medline_id.empty_values = medline_id.empty_values + [0]
    pubmed_id.empty_values = pubmed_id.empty_values + [0]

    def __str__(self):
        s = f'{self.cit_key[:30]}'
        if self.pubmed_id:
            s += f' (PM:{self.pubmed_id})'
        return s

    @classmethod
    def load(cls):
        rest = super().load()
        cit_id_field = cls.get_here_field()  # should be cit_id

        # expect rest to just be the citation <-> node m2m data

        citations = cls.objects.in_bulk(field_name=cit_id_field.name)
        qs = TaxNode.objects.values_list('taxid', 'pk')
        node_id_map = dict(qs.iterator())  # a lookup map taxid -> node pk

        for cit_id, dat in rest:
            node_field = cls._meta.get_field('node')
            values = dat[node_field].split()  # taxids
            node_pks = [node_id_map[int(i)] for i in values]
            cit_id = cit_id_field.to_python(cit_id)  # cit_id was str until now
            obj = citations[cit_id]
            obj.node.add(*node_pks)


class DeletedNode(NCBITaxModel):
    taxid = models.PositiveIntegerField(
        unique=True,
        help_text='deleted node id',
    )

    source_file = 'delnodes.dmp'

    def __str__(self):
        return str(self.taxid)


class Division(NCBITaxModel):
    division_id = models.PositiveIntegerField(
        help_text='taxonomy database division id',
        unique=True,
    )
    cde = models.CharField(
        max_length=3,
        help_text='GenBank division code (three characters)',
    )
    name = models.CharField(
        max_length=3,
    )
    comments = models.CharField(max_length=1000)

    source_file = 'division.dmp'

    def __str__(self):
        return str(self.name)


class Gencode(NCBITaxModel):
    genetic_code_id = models.PositiveIntegerField(
        help_text='GenBank genetic code id',
        unique=True,
    )
    abbreviation = models.CharField(
        max_length=10,
        help_text='genetic code name abbreviation',
    )
    name = models.CharField(
        max_length=64,
        unique=True,
        help_text='genetic code name',
    )
    cde = models.CharField(
        max_length=64,
        help_text='translation table for this genetic code',
    )
    starts = models.CharField(
        max_length=64,
        help_text='start codons for this genetic code',
    )

    source_file = 'gencode.dmp'


class Host(NCBITaxModel):
    node = models.ForeignKey('TaxNode', **fk_req)  # 1-1?
    potential_hosts = models.CharField(
        # FIXME: maybe make this m2m?
        max_length=32,
        help_text="theoretical host list separated by comma ','",
    )

    source_file = 'host.dmp'

    def __str__(self):
        return f'{self.node}/{self.potential_hosts}'


class MergedNodes(NCBITaxModel):
    old_taxid = models.PositiveIntegerField(
        unique=True,
        help_text='id of nodes which has been merged',
    )
    new_node = models.ForeignKey(
        'TaxNode',
        **fk_req,
        help_text='node which is result of merging',
    )

    source_file = 'merged.dmp'

    def __str__(self):
        return f'{self.old_taxid}->{self.new_node}'


class TaxName(NCBITaxModel):
    NAME_CLASS_SCI = 11
    NAME_CLASSES = (
        (1, 'acronym'),
        (2, 'authority'),
        (3, 'blast name'),
        (4, 'common name'),
        (5, 'equivalent name'),
        (6, 'genbank acronym'),
        (7, 'genbank common name'),
        (8, 'genbank synonym'),
        (9, 'in-part'),
        (10, 'includes'),
        (NAME_CLASS_SCI, 'scientific name'),
        (12, 'synonym'),
    )

    node = models.ForeignKey(
        'TaxNode',
        **fk_req,
        help_text='the node associated with this name',
    )
    name = models.CharField(max_length=128, help_text='the name itself')
    unique_name = models.CharField(
        max_length=128,
        help_text='the unique variant of this name if name not unique',
    )
    name_class = models.PositiveSmallIntegerField(
        choices=NAME_CLASSES,
        help_text='synonym, common name, ...',
    )

    source_file = 'names.dmp'

    class Meta:
        unique_together = (
            # don't need the unique_name? Ha!
            ('node', 'name', 'name_class'),
        )
        indexes = [
            models.Index(fields=['node', 'name_class']),
        ]

    def __str__(self):
        return self.name


class TaxNode(NCBITaxModel):
    taxid = models.PositiveIntegerField(
        unique=True,
        verbose_name='taxonomy ID',
    )
    parent = models.ForeignKey('self', **fk_opt, related_name='children')
    rank = models.CharField(max_length=32, db_index=True)
    embl_code = models.CharField(max_length=2, **ch_opt)
    division = models.ForeignKey(Division, **fk_req)
    is_div_inherited = models.BooleanField()
    gencode = models.ForeignKey(
        Gencode,
        **fk_req,
        related_name='node',
    )
    is_gencode_inherited = models.BooleanField()
    mito_gencode = models.ForeignKey(
        Gencode,
        **fk_req,
        related_name='node_mito',
    )
    is_mgc_inherited = models.BooleanField(
        help_text='node inherits mitochondrial gencode from parent',
    )
    is_genbank_hidden = models.BooleanField(
        help_text='name is suppressed in GenBank entry lineage',
    )
    hidden_subtree_root = models.BooleanField(
        help_text='this subtree has no sequence data yet',
    )
    comments = models.CharField(max_length=1000)
    plastid_gencode = models.ForeignKey(
        Gencode,
        **fk_opt,
        related_name='node_plastid',
    )
    is_pgc_inherited = models.NullBooleanField(**opt)
    has_specified_species = models.BooleanField()
    hydro_gencode = models.ForeignKey(
        Gencode,
        **fk_opt,  # missing in single row
        related_name='node_hydro',
    )
    is_hgc_inherited = models.BooleanField()

    source_file = 'nodes.dmp'

    class Meta:
        verbose_name = 'NCBI taxon'
        verbose_name_plural = 'ncbi taxa'

    def __str__(self):
        return f'{self.taxid}'

    @property
    def name(self):
        """
        Get the scientific name of node

        This works because (and as long as) each node has exactly one
        scientific name
        """
        return self.taxname_set.get(name_class=TaxName.NAME_CLASS_SCI)

    def is_root(self):
        """ Say if node is the root of the taxonomic tree """
        return self.taxid == 1

    STANDARD_RANKS = (
        'superkingdom', 'phylum', 'class', 'order', 'family', 'genus',
        'species',
    )

    @classmethod
    def load(cls):
        rest = super().load()

        # get nodes
        objs = cls.objects.in_bulk(field_name='taxid')

        # set parents
        parent_field = cls._meta.get_field('parent')
        for taxid, dat in rest:
            taxid = int(taxid)
            parent_taxid = int(dat[parent_field])
            parent = objs[parent_taxid]
            objs[taxid].parent = parent

        # save
        cls.objects.bulk_update(objs.values(), ['parent'])

    def lineage_list(self, full=True, names=True):
        lineage = list(reversed(list(self.ancestors(all_ranks=full))))
        if names:
            f = dict(node_id__in=lineage, name_class=TaxName.NAME_CLASS_SCI)
            qs = TaxName.objects.filter(**f)
            name_map = dict(qs.values_list('node__taxid', 'name'))
            lineage = [(i, name_map[i.taxid]) for i in lineage]

        return lineage

    def lineage(self, full=True):
        lin = [
            f'{i.rank}:{name}'
            for i, name in self.lineage_list(full=full, names=True)
        ]
        return ';'.join(lin)

    def ancestors(self, all_ranks=True):
        """
        Generate a node's ancestor nodes starting with itself, towards the root

        Will always yield self even if all_ranks is False.
        """
        cur = self
        while True:
            if all_ranks or cur.rank in self.STANDARD_RANKS or cur is self:
                yield cur
            if cur.is_root():
                break
            cur = cur.parent

    def lca(self, other, all_ranks=True):
        seen_a = set()
        seen_b = set()
        a_anc = self.ancestors(all_ranks=all_ranks)
        b_anc = other.ancestors(all_ranks=all_ranks)
        for a, b in zip_longest(a_anc, b_anc):
            if a is not None:
                seen_a.add(a)
            if b is not None:
                seen_b.add(b)

            if a in seen_b:
                return a
            if b in seen_a:
                return b

        # all_ranks == False but lca is above superkingdom
        return None

    def __eq__(self, other):
        # It's taxid or nothing
        return self.taxid == other.taxid

    def is_ancestor_of(self, other):
        """
        Check if self is ancestor of other

        This method is the base of the rich comparison method
        """
        # TODO: optimize by checking on rank?
        for i in other.ancestors(all_ranks=True):
            if self == i:
                return True
        return False

    def __lt__(self, other):
        return self.is_ancestor_of(other) and not self == other

    def __le__(self, other):
        return self.is_ancestor_of(other)

    def __gt__(self, other):
        return other.is_ancestor_of(self) and not self == other

    def __ge__(self, other):
        return other.is_ancestor_of(self)

    @classmethod
    def search(cls, query, any_name_class=False):
        """ convenience method to get nodes from a (partial) name """
        f = {}
        if not any_name_class:
            f['taxname__name_class'] = TaxName.NAME_CLASS_SCI
        if query[0].isupper():
            f['taxname__name__startswith'] = query
        else:
            f['taxname__name__icontains'] = query
        return cls.objects.filter(**f)  # FIXME: distinct()?


class TypeMaterial(NCBITaxModel):
    node = models.ForeignKey(TaxNode, **fk_req)
    tax_name = models.CharField(
        max_length=128,
        help_text='organism name type material is assigned to',
    )
    material_type = models.ForeignKey(
        'TypeMaterialType',
        **fk_req,
    )
    identifier = models.CharField(
        max_length=32,
        help_text='identifier in type material collection',
    )

    source_file = 'typematerial.dmp'

    def __str__(self):
        return f'{self.node}/{self.material_type}'


class TypeMaterialType(NCBITaxModel):
    name = models.CharField(
        max_length=64,
        unique=True,
        help_text='name of type material type',
    )
    synonyms = models.CharField(
        max_length=128,
        help_text='alternative names for type material type',
    )
    nomenclature = models.CharField(
        max_length=2,
        help_text='Taxonomic Code of Nomenclature coded by a single letter',
        # B - International Code for algae, fungi and plants (ICN), previously
        # Botanical Code, P - International Code of Nomenclature of Prokaryotes
        # (ICNP), Z - International Code of Zoological Nomenclature (ICZN), V -
        # International Committee on Taxonomy of Viruses (ICTV) virus
        # classification.
    )
    description = models.CharField(
        max_length=1024,
        help_text='descriptive text',
    )

    source_file = 'typeoftype.dmp'

    def __str__(self):
        return f'{self.name}'
