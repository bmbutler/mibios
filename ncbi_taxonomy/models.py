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
    name_class = models.CharField(
        max_length=32,
        db_index=True,
        help_text='synonym, common name, ...',
    )

    source_file = 'names.dmp'

    class Meta:
        unique_together = (
            # don't need the unique_name? Ha!
            ('node', 'name', 'name_class'),
        )

    def __str__(self):
        return self.name


class TaxNode(NCBITaxModel):
    taxid = models.PositiveIntegerField(
        unique=True,
        verbose_name='taxonomy ID',
    )
    parent = models.ForeignKey('self', **fk_opt)
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

    def lineage_list(self, names=True):
        lineage = []
        cur = self
        while True:
            if cur.taxid == 1:
                break
            lineage.append(cur)
            cur = cur.parent

        lineage = list(reversed(lineage))
        if names:
            f = dict(node__in=lineage, name_class='scientific name')
            qs = TaxName.objects.filter(**f)
            name_map = dict(qs.values_list('node__taxid', 'name'))
            lineage = [(i, name_map[i.taxid]) for i in lineage]

        return lineage

    def lineage(self):
        lin = [
            f'{i.rank}:{name}'
            for i, name in self.lineage_list(names=True)
        ]
        return ';'.join(lin)


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
