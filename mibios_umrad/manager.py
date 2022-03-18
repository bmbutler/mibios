from collections import defaultdict
from itertools import groupby, islice, zip_longest
from logging import getLogger
from operator import itemgetter, length_hint
from pathlib import Path
import os

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db.models.manager import Manager as DjangoManager
from django.db.transaction import atomic, set_rollback
from django.utils.module_loading import import_string

from mibios.models import (
    BaseManager as MibiosBaseManager,
    QuerySet as MibiosQuerySet,
)

from .utils import CSV_Spec, ProgressPrinter, siter


log = getLogger(__name__)


class BulkCreateWrapperMixin:
    @staticmethod
    def bulk_create_wrapper(wrappee_bc):
        """
        A wrapper to add batching and progress metering to bulk_create()

        :param bound method wrappee_bc: A create_bulk method bound to a manager

        This static wrapper method allows us to still (1) override the usual
        Manager.bulk_create() but also (2) offer a wrapper to be used for cases
        in which we are unable or don't want to override the model's manager,
        e.g. the implicit through model of a many-to-many relation.
        """
        def bulk_create(objs, batch_size=None, ignore_conflicts=False,
                        progress_text=None, progress=True):
            """
            Value-added bulk_create

            Does not return the object list like super().bulk_create() does.
            """
            if not progress:
                wrappee_bc(
                    objs,
                    ignore_conflicts=ignore_conflicts,
                    batch_size=batch_size,
                )
                return

            if batch_size is None:
                # sqlite has a variable-per-query limit of 999; it's not clear
                # how one runs against that; it seems that multiple smaller
                # INSERTs are being used automatically.  So, until further
                # notice, only have a default batch size for progress metering
                # here.
                batch_size = 999

            # get model name from manager (wrappee_bc.__self__ is the manager)
            model_name = wrappee_bc.__self__.model._meta.verbose_name

            if progress_text is None:
                progress_text = f'{model_name} records created'

            objs = iter(objs)
            pp = ProgressPrinter(
                progress_text,
                length=length_hint(objs) or None,
            )

            while True:
                batch = list(islice(objs, batch_size))
                if not batch:
                    break
                try:
                    wrappee_bc(
                        batch,
                        ignore_conflicts=ignore_conflicts,
                    )
                except Exception as e:
                    print(f'ERROR saving to {model_name or "?"}: batch 1st: '
                          f'{vars(batch[0])=}')
                    raise RuntimeError('error saving batch', batch) from e
                pp.inc(len(batch))

            pp.finish()

        return bulk_create

    def bulk_create(self, objs, batch_size=None, ignore_conflicts=False,
                    progress_text=None, progress=True):
        """
        Value-added bulk_create with batching and progress metering

        Does not return the object list like super().bulk_create() does.
        """
        wrapped_bc = self.bulk_create_wrapper(super().bulk_create)
        wrapped_bc(
            objs=objs,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
            progress_text=None,
            progress=True,
        )


class Loader(BulkCreateWrapperMixin, DjangoManager):
    """
    A manager providing functionality to load data from file
    """
    def get_file(self):
        """
        Return pathlib.Path to input data file
        """
        raise NotImplementedError('must be implemented in inheriting model')

    spec = None
    """
    A list of field names or list of tuples matching file headers to field
    names.  Use list of tuples if import file has a header.  Otherwise use the
    simple list, listing the fields in the right order.
    """

    def contribute_to_class(self, model, name):
        super().contribute_to_class(model, name)
        # get spec from model if none is declared in loader class
        if self.spec is None:
            try:
                self.spec = model.loader_spec
            except AttributeError:
                pass

        if self.spec is not None:
            if self.spec.model is None:
                self.spec.model = model

            self.spec.loader = self
            self.spec.setup()

    def load(self, max_rows=None, start=0, dry_run=False, sep='\t',
             parse_only=False, file=None, template={}):
        """
        Load data from file

        :param int start:
            0-based input file line (not counting any header) from which to
            start loading, equivalent to number of non-header lines skipped.
        :param bool parse_only:
            Return each row as a dict and don't use the general loader logic.

        May assume empty table ?!?
        """
        # ensure file is a Path
        if file is None:
            file = self.get_file()
        elif isinstance(file, str):
            file = Path(file)

        with file.open() as f:
            print(f'File opened: {f.name}')
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            if self.spec is None:
                raise NotImplementedError(f'{self}: loader spec not set')

            if self.spec.has_header:
                # check header
                head = f.readline().rstrip('\n').split(sep)
                cols = self.spec.all_cols
                for i, (a, b) in enumerate(zip(head, cols), start=1):
                    # ignore column number differences here, will catch later
                    if a != b:
                        raise ValueError(
                            f'Unexpected header row: first mismatch in column '
                            f'{i}: got "{a}", expected "{b}"'
                        )
                if len(head) != len(self.spec):
                    raise ValueError(
                        f'expecting {len(self.spec)} columns but got '
                        f'{len(head)} in header row'
                    )
            else:
                # assume no header
                pass

            pp = ProgressPrinter(f'{self.model._meta.model_name} records read')
            f = pp(f)

            if max_rows is None and start == 0:
                file_it = f
            else:
                file_it = islice(f, start, start + max_rows)

            if parse_only:
                return self._parse_lines(file_it, sep=sep)

            return self._load_lines(
                file_it,
                template=template,
                sep=sep,
                dry_run=dry_run,
            )

    @atomic
    def _load_lines(self, lines, sep='\t', dry_run=False, template={}):
        ncols = len(self.spec)
        split = self._split_m2m_input
        fields = [self.model._meta.get_field(i) for i in self.spec.keys]
        convfuncs = self.spec.convfuncs
        cut = self.spec.cut
        empty_extra = self.spec.empty_values

        # loading FK (acc,...)->pk mappings
        fkmap = {}
        for i in fields:
            if not i.many_to_one:
                continue
            print(f'Loading {i.related_model._meta.verbose_name} data...',
                  end='', flush=True)
            lookups = i.related_model.get_accession_lookups()
            fkmap[i.name] = {
                tuple(a): pk for *a, pk
                in i.related_model.objects.values_list(*lookups, 'pk')
                    .iterator()
            }
            print('[OK]')

        objs = []
        m2m_data = {}
        missing_fks = defaultdict(set)
        skip_count = 0
        pk = None

        for line in lines:
            obj = self.model(**template)
            m2m = {}
            row = line.rstrip('\n').split(sep)

            if len(row) != ncols:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {row=} '
                    f'{ncols=}'
                )

            for field, fn, value in zip(fields, convfuncs, cut(row)):
                if callable(fn):
                    value = fn(value, row)
                    if value is CSV_Spec.IGNORE_COLUMN:
                        continue  # treats value as empty
                    elif value is CSV_Spec.SKIP_ROW:
                        break  # skips obj / avoids for-else block

                if field.many_to_many:
                    m2m[field.name] = split(value)
                elif field.many_to_one:
                    if not isinstance(value, tuple):
                        value = (value, )  # fkmap keys are tuples
                    try:
                        pk = fkmap[field.name][value]
                    except KeyError:
                        # FK target object does not exist
                        missing_fks[field.name].add(value)
                        if field.null:
                            pass
                        else:
                            skip_count += 1
                            break  # will skip this obj / skips for-else block
                    else:
                        setattr(obj, field.name + '_id', pk)
                elif value not in empty_extra and value not in field.empty_values:  # noqa: E501
                    # TODO: find out why leaving '' in for int fields fails
                    # ValueError @ django/db/models/fields/__init__.py:1825
                    setattr(obj, field.name, value)
                else:
                    # empty field
                    pass
            else:
                objs.append(obj)
                if m2m:
                    m2m_data[obj.get_accessions()] = m2m
        del fkmap, field, value, row, pk, line, obj, m2m

        for fname, bad_ids in missing_fks.items():
            print(f'WARNING: found {len(bad_ids)} distinct unknown {fname}'
                  'IDs:', ' '.join([str(i) for i in islice(bad_ids, 5)]))
        if skip_count:
            print(f'WARNING: skipped {skip_count} rows due to unknown but '
                  f'non-null FK IDs')

        if not objs:
            print('WARNING: nothing saved, empty file or all got skipped')
            return

        if missing_fks:
            del fname, bad_ids
        del missing_fks, skip_count

        self.bulk_create(objs)
        del objs

        if m2m_data:
            # get accession -> pk map
            qs = self.model.objects.values_list(
                *self.model.get_accession_lookups(),
                'pk',
            )
            accpk = {tuple(accs): pk for *accs, pk in qs.iterator()}

            # replace accession with pk in m2m data keys
            m2m_data = {accpk[i]: data for i, data in m2m_data.items()}
            del accpk

            # collecting all m2m entries
            for field in (i for i in fields if i.many_to_many):
                self._update_m2m(field.name, m2m_data)

        set_rollback(dry_run)

    def _update_m2m(self, field_name, m2m_data):
        """
        Update M2M data for one field -- helper for _load_lines()

        :param str field_name: Name of m2m field
        :param dict m2m_data:
            A dict with all fields' m2m data as produced in the load_lines
            method.
        :param callable create_fun:
            Callback function that creates a missing instance on the other side
            of the relation.
        """
        print(f'm2m {field_name}: ', end='', flush=True)
        field = self.model._meta.get_field(field_name)
        model = field.related_model

        # extract and flatten all accessions for field in m2m data
        accs = (i for objdat in m2m_data.values() for i in objdat[field_name])
        acc_field = model.get_accession_field_single()
        if acc_field.get_internal_type().endswith('IntegerField'):
            # cast to right type as needed (integers only?)
            accs = (int(i) for i in accs)
        accs = set(accs)
        print(f'{len(accs)} unique accessions in data - ', end='', flush=True)
        if not accs:
            print()
            return

        # get existing
        qs = model.objects.all()
        qs = qs.values_list(model.get_accession_lookup_single(), 'pk')
        a2pk = dict(qs.iterator())
        print(f'known: {len(a2pk)} ', end='', flush=True)

        new_accs = [i for i in accs if i not in a2pk]
        if new_accs:
            # save additional remote side objects
            # NOTE: this will only work for those models for which the supplied
            # information (accession, source model and field) is sufficient,
            # might require overriding create_from_m2m_input().
            print(f'new: {len(new_accs)}', end='')
            if a2pk:
                # if old ones exist, then display some of the new ones, this
                # is for debugging, in case these ought to be known
                print(' ', ', '.join([str(i) for i in islice(new_accs, 5)]),
                      '...')
            else:
                print()

            model.objects.create_from_m2m_input(
                new_accs,
                source_model=self.model,
                src_field_name=field_name,
            )

            # get m2m field's key -> pk mapping
            a2pk = dict(qs.iterator())
        else:
            print()

        # set relationships
        rels = []  # pairs of ours and other's PKs
        for i, other in m2m_data.items():
            rels.extend((
                (i, a2pk[acc_field.to_python(j)])
                for j in other[field_name]
            ))
        Through = field.remote_field.through  # the intermediate model
        our_id_name = self.model._meta.model_name + '_id'
        other_id_name = model._meta.model_name + '_id'
        through_objs = [
            Through(
                **{our_id_name: i, other_id_name: j}
            )
            for i, j in rels
        ]
        self.bulk_create_wrapper(Through.objects.bulk_create)(through_objs)

    def _split_m2m_input(self, value):
        """
        Helper to split semi-colon-separated list-field values in import file
        """
        # split and remove empties:
        items = (i for i in value.split(';') if i)
        # TODO: duplicates in input data (NAME/function column), tell Teal?
        # TODO: check with Teal if order matters or if it's okay to sort
        items = sorted(set(items))
        return items

    def _parse_lines(self, lines, sep='\t'):
        ncols = len(self.spec)

        data = []
        split = self._split_m2m_input

        for line in lines:
            row = line.rstrip('\n').split(sep)

            if len(row) != ncols:
                raise ValueError(
                    f'bad num of cols ({len(row)}), {row=} '
                    f'{ncols=}'
                )

            rec = {}
            for key, value in zip(self.spec.all_keys, row):
                if key is None:
                    continue
                try:
                    field = self.model._meta.get_field(key)
                except FieldDoesNotExist:
                    # caller must handle values from m2m field (if any)
                    pass
                else:
                    if field.many_to_many:
                        value = split(value)

                rec[key] = value

            data.append(rec)
        return data


class CompoundLoader(Loader):
    """ loader for compound data from all_compound_info file """

    def get_file(self):
        return settings.UMRAD_ROOT \
            / f'all_compound_info_{settings.UMRAD_VERSION}.txt'

    @atomic
    def load(self, max_rows=None, dry_run=False):
        # get related model classes here to avoid circular imports
        CompoundEntry = import_string('mibios_umrad.models.CompoundEntry')
        CompoundName = import_string('mibios_umrad.models.CompoundName')

        refdb_names = [
            i for i, _ in CompoundEntry.DB_CHOICES
            if i in self.spec.keys
        ]

        # get data and split m2m fields
        data = []
        for row in super().load(max_rows=max_rows, parse_only=True):
            for i in refdb_names + ['names']:
                row[i] = self._split_m2m_input(row[i])
            data.append(row)

        unicomps = []  # collector for unique compounds (this model)
        name_data = {}  # accession/id to names association

        grpk = itemgetter(*refdb_names)  # sort/group by related IDs columns
        data = sorted(data, key=grpk)
        pp = ProgressPrinter('compound records processed')
        data = pp(data)
        extra_ids = set()
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
                grp_ids = [i['accession'] for i in grp]
                extra_ids.update(all_ids.difference(grp_ids))
                del grp_ids

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
                name_data[acc] = row['names']

            unicomps.append(cgroup)

        if extra_ids:
            print(f'WARNING: Found {len(extra_ids)} distinct compound IDs that'
                  ' didn\'t have a row in their group) that will be ignored:',
                  ', '.join([str(i) for i in islice(extra_ids, 5)]),
                  '...')

        # create Compound objects and get PKs
        objs = (self.model() for _ in range(len(unicomps)))
        self.bulk_create(objs, batch_size=500)
        pks = self.values_list('pk', flat=True)

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
        rels = (  # the comp->name relation
            (comp_pk, name_pk_map[name_entry])
            for acc, comp_pk in pk_map.items()
            for name_entry in name_data[acc]
        )
        through = CompoundEntry._meta.get_field('names').remote_field.through
        through_objs = [
            through(
                **{'compoundentry_id': i, 'compoundname_id': j}
            )
            for i, j in rels
        ]
        self.bulk_create_wrapper(through.objects.bulk_create)(through_objs)

        if dry_run:
            set_rollback(True)


class FuncRefDBEntryLoader(Loader):
    spec = CSV_Spec('accession', 'names')

    def get_file(self):
        return settings.UMRAD_ROOT / \
            f'Function_Names_{settings.UMRAD_VERSION}.txt'


class LineageLoader(Loader):
    """ loader for lineage model """

    @atomic
    def load(self, path=None, dry_run=False):
        """
        Upload data to table

        Returns list of (taxid, pk) tuples.
        Method should be called from Taxon._load()
        """
        TaxName = import_string('mibios_umrad.models.TaxName')
        rows = TaxName.loader.load(path)
        name2pk = {
            (i, j): k for i, j, k
            in TaxName.objects.values_list('name', 'rank', 'pk').iterator()
        }
        rankid = {
            j: i for i, j in TaxName.RANKS if j != 'root'
        }
        lin2taxids = defaultdict(list)  # maps name PK tuples to list of taxids
        objs = []
        fields = self.model.get_name_fields()
        for row in rows:
            obj = self.model()
            # NOTE: row may have variable length
            for f, val in zip_longest(fields, row[1:]):
                if val:
                    pk = name2pk[(val, rankid[f.name])]
                    setattr(obj, f.name + '_id', pk)

            if obj.get_name_pks() not in lin2taxids:
                # first time seen
                objs.append(obj)

            lin2taxids[obj.get_name_pks()].append(row[0])

        self.bulk_create(objs)
        del objs

        # return one to many mapping taxid -> lineage PK
        tid2pk = {
            tid: obj.pk
            for obj in self.all().iterator()
            for tid in lin2taxids[obj.get_name_pks()]
        }
        set_rollback(dry_run)
        return tid2pk


class ReactionLoader(Loader):
    """ loader for reaction data from all_reaction_info file """

    def get_file(self):
        return settings.UMRAD_ROOT \
            / f'all_reaction_info_{settings.UMRAD_VERSION}.txt'

    @atomic
    def load(self, max_rows=None, dry_run=False):
        # get related model classes here to avoid circular imports
        Compound = import_string('mibios_umrad.models.Compound')
        CompoundEntry = import_string('mibios_umrad.models.CompoundEntry')
        ReactionEntry = import_string('mibios_umrad.models.ReactionEntry')

        refdbs = [
            i for i, _ in ReactionEntry.DB_CHOICES
            if i in self.spec.keys
        ]
        comp_cols = {
            ReactionEntry.DB_KEGG: ('left_kegg', 'right_kegg'),
            ReactionEntry.DB_BIOCYC: ('left_biocyc', 'right_biocyc'),
            ReactionEntry.DB_RHEA: ('left_rhea', 'right_rhea'),
        }

        # get data and split m2m fields
        data = []
        m2mcols = [i for i in self.spec.all_keys if i not in ['accession', 'dir']]  # noqa:E501
        for row in super().load(max_rows=max_rows, parse_only=True):
            for i in m2mcols:
                row[i] = self._split_m2m_input(row[i])
            data.append(row)

        urxns = []

        # sort/group and process by reaction group
        grpk = itemgetter(*refdbs)
        data = sorted(data, key=grpk)
        pp = ProgressPrinter('reaction entry records processed')
        data = pp(data)
        extra_ids = set()
        compounds = {}
        for key, grp in groupby(data, grpk):
            grp = list(grp)
            all_ids = {i for j in key for i in j}  # flatten list-of-list
            rxngroup = []

            if len(grp) < len(all_ids):
                # extra IDs / missing a row?  See similar check in Compounds
                grp_ids = [i['accession'] for i in grp]
                extra_ids.update(all_ids.difference(grp_ids))
                del grp_ids

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

        if extra_ids:
            print(f'WARNING: Found {len(extra_ids)} distinct reaction IDs that'
                  ' didn\'t have a row in their group) that will be ignored:',
                  ', '.join([str(i) for i in islice(extra_ids, 5)]),
                  '...')

        # create Reaction objects and get PKs
        objs = (self.model() for _ in range(len(urxns)))
        self.bulk_create(objs, batch_size=500)
        pks = self.values_list('pk', flat=True)

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
                  'reaction data:',
                  ', '.join([str(i) for i in islice(unknown_cpd_accs, 5)]),
                  '...')
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
            through_objs = [
                through(**{'reactionentry_id': i, 'compoundentry_id': j})
                for i, j in rels
            ]
            bc = self.bulk_create_wrapper(through.objects.bulk_create)
            bc(through_objs)

        set_rollback(dry_run)


class TaxNameLoader(Loader):
    """ data loader for the TaxName model """

    @atomic
    def load(self, path=None, dry_run=False):
        """
        Reads TAXONOMY_DB and populates TaxName

        Returns the rows for further processing.  Method should be called via
        Taxon.load().
        """
        if path is None:
            path = self.get_file()

        data = []
        with path.open() as f:
            print(f'Reading from file {path} ...')
            f = ProgressPrinter('taxa found')(f)
            log.info(f'reading taxonomy: {path}')
            for line in f:
                data.append(line.strip().split('\t'))

        pp = ProgressPrinter('tax names processed')
        rankids = [i[0] for i in self.model.RANKS[1:]]
        names = dict()
        for row in data:
            for rid, name in zip_longest(rankids, row[1:]):
                if rid is None:
                    raise RuntimeError(f'unexpectedly low ranks: {row}')
                if name is None:
                    # no strain etc
                    continue
                key = (rid, name)
                if key not in names:
                    names[key] = self.model(rank=rid, name=name)
                    pp.inc()
        pp.finish()

        log.info(f'Storing {len(names)} unique tax names to DB...')
        self.bulk_create(names.values())
        set_rollback(dry_run)
        return data

    @classmethod
    def get_file(cls):
        return settings.UMRAD_ROOT / \
            f'TAXONOMY_DB_{settings.UMRAD_VERSION}.txt'


class TaxonLoader(Loader):
    """ loader for Taxon model """

    @atomic
    def load(self, path=None, dry_run=False):
        Lineage = import_string('mibios_umrad.models.Lineage')
        taxid2linpk = Lineage.loader.load(path)

        self.bulk_create(siter((
            self.model(taxid=i, lineage_id=j)
            for i, j in taxid2linpk.items()
        ), len(taxid2linpk)))

        set_rollback(dry_run)


class BaseManager(BulkCreateWrapperMixin, MibiosBaseManager):
    """ Manager class for UMRAD data models """
    def create_from_m2m_input(self, values, source_model, src_field_name):
        """
        Store objects from accession or similar value (and context, as needed)

        :param list values:
            List of accessions
        :param Model source_model:
            The calling model, the model on the far side of the m2m relation.
        :param str src_field_name:
            Name of the m2m field pointing to us

        Inheriting classes should overwrite this method if more that the
        accession or other unique ID field is needed to create the instances.
        The basic implementation will assign the given value to the accession
        field, if one exists, or to the first declared unique field, other
        that the standard id AutoField.

        This method is responsible to set all required fields of the model,
        hence the default version should only be used with controlled
        vocabulary or similarly simple models.
        """
        # TODO: if need arises we can implement support for values to
        # contain tuples of values
        attr_name = self.model.get_accession_lookup_single()
        model = self.model
        objs = (model(**{attr_name: i}) for i in values)
        return self.bulk_create(objs)


class QuerySet(MibiosQuerySet):
    def search(self, query_term, field_name=None, lookup=None):
        """
        implement search from search field

        For models for which get_accession_field_single raises an exception,
        this method should be overwritten.
        """
        if field_name is None:
            uid_field = self.model.get_search_field()
        else:
            uid_field = self.model._meta.get_field(field_name)

        if lookup is None:
            if uid_field.get_internal_type() == 'CharField':
                lookup = 'icontains'
            else:
                lookup = 'exact'

        kw = {uid_field.name + '__' + lookup: query_term}
        try:
            return self.filter(**kw)
        except ValueError:
            # wrong type, e.g. searching for "text" on a numeric field
            return self.none()


class Manager(BaseManager.from_queryset(QuerySet)):
    pass
