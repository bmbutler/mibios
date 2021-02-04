from collections import OrderedDict
from itertools import chain, groupby
from operator import attrgetter, itemgetter

from Bio import SeqIO
from django.db import models
from django.db.transaction import atomic

from omics.shared import MothurShared
from mibios.dataset import UserDataError
from mibios.models import (ImportFile, Manager, CurationManager, Model,
                           ParentModel, QuerySet, TagNote)
from mibios.utils import getLogger


log = getLogger(__name__)


class Sample(ParentModel):
    """
    Parent model for samples

    This is the multi-table-inheritance parent that other apps should use to
    interface with sequencing data.  There are no fields declared here besides
    the usual auto-primary-key and history.
    """
    pass


class Sequencing(Model):
    MOCK = 'mock'
    WATER = 'water'
    BLANK = 'blank'
    PLATE = 'plate'
    OTHER = 'other'
    CONTROL_CHOICES = (
        (MOCK, MOCK),
        (WATER, WATER),
        (BLANK, BLANK),
        (PLATE, PLATE),
        (OTHER, OTHER),
    )
    name = models.CharField(max_length=100, unique=True)
    sample = models.ForeignKey(Sample, on_delete=models.SET_NULL,
                               blank=True, null=True)
    control = models.CharField(max_length=50, choices=CONTROL_CHOICES,
                               blank=True)
    r1_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    r2_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    note = models.ManyToManyField(TagNote, blank=True)
    run = models.ForeignKey('SequencingRun', on_delete=models.CASCADE,
                            blank=True, null=True)
    plate = models.PositiveSmallIntegerField(blank=True, null=True)
    plate_position = models.CharField(max_length=10, blank=True)
    snumber = models.PositiveSmallIntegerField(blank=True, null=True)
    otu = models.ManyToManyField(
        'OTU',
        through='Abundance',
        editable=False,
    )

    class Meta:
        unique_together = (
            ('run', 'snumber'),
            ('run', 'plate', 'plate_position'),
        )
        ordering = ['name']

    @classmethod
    def parse_control(cls, txt):
        """
        Coerce text into available control choices
        """
        choice = txt.strip().lower()
        if choice:
            for i in (j[0] for j in cls.CONTROL_CHOICES):
                if i in choice:
                    return i
            return cls.OTHER
        else:
            return ''


class SequencingRun(Model):
    serial = models.CharField(max_length=50)
    number = models.PositiveSmallIntegerField()
    path = models.CharField(max_length=2000, blank=True)

    class Meta:
        unique_together = ('serial', 'number')
        ordering = ['serial', 'number']

    @Model.natural.getter
    def natural(self):
        return '{}-{}'.format(self.serial, self.number)

    @classmethod
    def natural_lookup(cls, value):
        if value is cls.NOT_A_VALUE:
            s, n = None, None
        else:
            s, n = value.split('-')
            n = int(n)
        return dict(serial=s, number=n)


class Sequence(Model):
    """
    Models a 16S/V4 sequence

    Can be an ASV or a representative sequence
    """
    taxon = models.ForeignKey('Taxonomy', on_delete=models.SET_NULL,
                              blank=True, null=True)
    seq = models.CharField(
        max_length=300,  # >> length of 16S V4
        unique=True,
        editable=False,
        verbose_name='sequence',
    )

    def __str__(self):
        return self.seq[:20] + '...'


class Strain(Model):
    sequence = models.ForeignKey(Sequence, on_delete=models.SET_NULL,
                                 blank=True, null=True)


class AbundanceQuerySet(QuerySet):
    _project = None

    def _clone(self):
        """
        Extends non-public _clone() to keep track of extra state
        """
        c = super()._clone()
        c._project = self._project
        return c

    def filter_project(self, project=None):
        """
        Ensure query set has only data for given project

        This method can be used to replace calls like:

            qs.filter(project=p)

        but does a bit more, it memorized the project, so repeated calls (with
        the sample project) don't filter again and can be used to ensure and
        assert that only on project's data is present in the data set without
        having to specify which.  Calling the method again but with a different
        project argument raises a ValueError.

        :param project: AnalysisProject instance or the name as str or None

        If the given project is None, then the query set is checked if it
        contains more than a single project's data, in which case a ValueError
        is raised.  This emits up to two additional and light SQL queries.  The
        method is idempotent in the sense that the returned query set remembers
        the project (accessible via the _project member variable) and
        subsequent calles just return a cloned instance.

        Returns a new QuerySet instance.
        """
        if self._project is not None:
            if project is not None and project != self._project:
                raise ValueError(
                    'AbundanceQuerySet is already filtered by another project')
            return self._clone()

        if isinstance(project, str):
            project = AnalysisProject.objects.get(name=project)
        elif project is None:
            project_pks = (self
                           .order_by('project')
                           .values_list('project', flat=True)
                           .distinct())
            if len(project_pks) == 0:
                # empty query set, leave project as None (what else can we do?)
                pass
            elif len(project_pks) == 1:
                project = AnalysisProject.objects.get(pk=project_pks[0])
            else:
                raise ValueError(
                    f'Abundance query set is related to {len(project_pks)} '
                    f'projects, need to know on which to filter, but project '
                    f'argument is None'
                )

        qs = self.filter(project=project)
        qs._project = project
        return qs

    def as_shared(self):
        """
        Make mothur-shared table

        Note: without label/numOtus columns

        Returns a pandas DataFrame.  Assumes, that the QuerySet is filtered to
        counts from a single analysis project but this is not checked.  If the
        assumption is violated, then the pivot operation will probably raise a:

            "ValueError: Index contains duplicate entries, cannot reshape"

        Missing counts are inserted as zero, mirroring the skipping of zeros at
        import.

        DEPRECATED (and possibly incorrect)
        """
        df = (
            self
            .as_dataframe('otu', 'sequencing', 'count', natural=True)
            .pivot(index='sequencing', columns='otu', values='count')
        )
        df.fillna(value=0, inplace=True)  # pivot introduced NaNs
        return df

    def as_shared_values_list_old(self):
        """
        Make mothur-shared table (slow, memory intense version)

        Returns an iterator over tuple rows, first row is the header.  This is
        intended to support data export.

        DEPRECATED (and possibly incorrect)
        """
        sh = self.as_shared()
        header = ['Group'] + list(sh.columns)
        recs = sh.itertuples(index=True, name=None)
        return chain([header], recs)

    @classmethod
    def _normalize(cls, group, size, debug=0):
        """
        Scale absolute counts to normal sample size
        """
        group = list(group)
        debug <= 1 or print(f'group={group}')
        vals = [i[0] for i in group]
        debug <= 1 or print(f'vals={vals}')
        total = sum(vals)
        debug <= 0 or print(f'total={total}')
        f = size / total
        frac = [i * f for i in vals]
        debug <= 1 or print(f'frac={frac}')
        disc = [round(i) for i in frac]
        overhang = sum(disc) - size
        debug <= 0 or print(f'overhang={overhang}')
        for _ in range(abs(overhang)):
            errs = [i - j for i, j in zip(disc, frac)]  # round-up errs are pos
            debug <= 1 or print(f'dict={disc} at {sum(disc)}')
            debug <= 1 or print(f'errs={errs}')
            if overhang > 0:
                abs_err_max = max(errs)
            else:
                abs_err_max = min(errs)
            idx = errs.index(abs_err_max)
            if overhang > 0:
                disc[idx] -= 1
            else:
                disc[idx] += 1
            debug <= 0 or print(f'idx={idx} count={disc[idx]} err: '
                                '{errs[idx]}->{disc[idx]-frac[idx]}')
        debug <= 1 or print(f'disc={disc} at {sum(disc)}')
        debug <= 1 or print(sum(disc))
        return ((i, j, k) for i, (_, j, k) in zip(disc, group))

    @classmethod
    def test_norm(cls, vals, norm_size, verbose=False, verboser=False):
        if verboser:
            debug = 2
        elif verbose:
            debug = 1
        else:
            debug = 0
        ret = cls._normalize(
            ((i, None, None) for i in vals),
            norm_size,
            debug=debug,
        )
        if not verboser:
            print([i[0] for i in ret])

    @classmethod
    def _unit_normalize(cls, group):
        """
        Normalize to unit interval, return fractions

        Deprecated

        This gets fractions from absolute counts, but is not needed if relative
        abundance is sorted in the DB.
        """
        group = list(group)
        vals = [i[0] for i in group]
        total = sum(vals)
        return ((i / total, j, k) for i, j, k in group)

    def _zerofill_and_norm(self, otus, group, normalize):
        """
        Fill in zeros and normalize values for one row

        This is a generator over the items of a row in a shared file

        :param otus: iterable over all OTU pks of QuerySet in order
        :param group: Data for one sample/row; an iterable over tuple
                      (abund, OTU pk)
        """
        if normalize == -1 or normalize == 0:
            zero = 0.0
        else:
            zero = 0

        abund, otu_pk = next(group)
        for i in otus:
            if otu_pk == i:
                yield abund
                try:
                    abund, otu_pk = next(group)
                except StopIteration:
                    pass
            else:
                yield zero

    def _group_and_pivot(self, otus, normalize, id_fields, min_avg_group_size):
        """
        Generator of shared file rows

        :param int normalize: Normalization mode:
            None -- absolute counts are returned
            -1   -- relative abundance in %
            0    -- relative abundance, as fractional values in
                    [0, 1] are returned.
            >= 1 -- then the counts are normalized by 'discrete scaling' to
                    the targeted sample size.  Note: values close to 1 are
                    useless, they result in relatively large quantization
                    errors.

        Assumes that filter_project() has been called if we are working with
        averaged data.
        """
        if normalize is None:
            abund_field = 'count'
        else:
            abund_field = 'relative'

        if self._avg_by:
            for i in id_fields:
                if not i.startswith('sequencing__'):
                    raise ValueError('expectiing id_fields to be related via '
                                     'Sequencing')
            # adjust lookups to be relative to Sequencing:
            seq_id_fields = [i[len('sequencing__'):] for i in id_fields]
            # get a map: row id -> real avg group count; this is a bit tricky
            # because we may have otherwise good Sequencing records without
            # abundance data (maybe filtered out by the analysis pipeline); the
            # avg must be calculated based on the records with abundance data.
            # The solution with Exist sub-queries seems to be reasonably fast:
            qs = Sequencing.curated.all()
            qs = qs.annotate(has_counts=models.Exists(
                Abundance.objects.filter(
                    sequencing=models.OuterRef('pk'),
                    project=self._project,
                )
            ))
            qs = qs.filter(has_counts=True)
            qs = qs.average(*seq_id_fields, natural=False)
            real_avg_grp_counts = {
                tuple([i[j] for j in seq_id_fields]): i['avg_group_count']
                for i in qs
            }
            del qs

        if self._avg_by:
            fields = [*id_fields, 'avg_group_count', abund_field, 'otu']
        else:
            fields = [*id_fields, abund_field, 'otu']

        it = (
            self.order_by(*id_fields, 'otu')
            .values_list(*fields)
            .iterator()
        )
        del fields

        # Group data by what will be the rows of the shared file.  The group-by
        # key, the row id(s) (a tuple) will be the first element of the output
        # row, the rest, which are now tuples of abundance value and otu, in
        # two steps: (1) possibly normalize the values and (2) fill in the
        # zeros.  Then that gets packaged into a row, each of which is yielded
        # back.
        key = itemgetter(slice(None, len(id_fields)))
        for row_id, group in groupby(it, key=key):
            # row_id is tuple of id_fields from above
            # a group is data for one row in shared file

            # pick only abund, otu pairs from data
            if self._avg_by:
                real_avg_g_ct = real_avg_grp_counts[row_id]
                if real_avg_g_ct < min_avg_group_size:
                    # skip this row
                    continue
                group = (
                    # correct averages for zeros!
                    (abund * (avg_g_ct / real_avg_g_ct), otu)
                    for *_, avg_g_ct, abund, otu in group
                )
                grp_count = (real_avg_g_ct,)
            else:
                # last two elements -> (abund, otu)
                group = map(itemgetter(slice(-2, None)), group)
                grp_count = ()

            if normalize is not None:
                if normalize == -1:
                    group = (
                        (abund * 100.0, otu)
                        for abund, otu in group
                    )
                elif normalize >= 1:
                    group = (
                        (round(abund * normalize), otu)
                        for abund, otu in group
                    )
                else:
                    # normalize == 0 -- keep values
                    pass

            values = self._zerofill_and_norm(otus, group, normalize)
            yield (row_id, grp_count, values)

    def _add_meta_data(self, shared_rows, meta_cols):
        """
        Replace row ids with natural keys and add extra meta-data columns

        :param iter shared_row: iterator over rows, should yield tuple of row
                                components, first component are the row ids, a
                                tuple of usually one Sequencing primary key,
                                or, for averaged rows, the pks of the models we
                                averaged by.
        :param list meta_cols: List of 2-tuples (Model, accessor) in the order
                               the columns will appear in the shared table.
                               For averaged data this must include the auto-
                               matic averaged-by id columns.  The row ids must
                               come first and have the *correct* order.  The
                               accessors must end with a field (which can be
                               "natural").

        Helper generator method for as_shared_values_list()
        """
        # get models for id/key columns, in same order.  Assumes those appear
        # first in meta_cols and other extra column use same models, so this
        # loop breaks at the first already seen model
        models = []
        for i, _ in meta_cols:
            if i in models:
                break
            models.append(i)

        # maps: one map per row_id column (or one per model)
        maps = []
        for m in models:
            # get parameters for select_related()
            related = []
            for mi, ai in meta_cols:
                if mi != m:
                    # different model
                    continue
                ai = ai.split('__')
                if len(ai) > 1:
                    # rm field at end of accessor
                    rel = '__'.join(ai[:-1])
                    if rel not in related:
                        related.append(rel)
                else:
                    # not a relation, just a local field
                    pass
            del mi, ai

            # mask accessors with '' for other models to keep correct order and
            # column positions, the getter will insert a None at those
            # position, maps for other models may provide values
            masked_cols = [j if i == m else '' for i, j in meta_cols]
            if m == Sequencing:
                qs = self._project.sequencing
            else:
                # s/objects/curated/ ?
                qs = m.objects.all()
            qs = qs.distinct().select_related(*related)
            idmap = {i.pk: i.getter(*masked_cols) for i in qs.iterator()}
            maps.append(idmap)
        del m

        for row_ids, grp_count, values in shared_rows:
            # combine values from different maps into one row
            meta_vals = [None] * len(meta_cols)
            for rid, m in zip(row_ids, maps):
                for i, j in enumerate(m[rid]):
                    if j is not None:
                        meta_vals[i] = j

            yield tuple(chain(meta_vals, grp_count, values))

    def as_shared_values_list(
            self,
            normalize=None,
            meta_cols=(),
            mothur=False,
            min_avg_group_size=1,
    ):
        """
        Make a "shared" table for download

        :param tuple meta_cols: Field accessors for meta data column.  With
        '__' separator.  However, must start with a model name of an id
        column's model.  For non-averaged data that is always 'sequencing'.
        For averaged data these are the model names of the averaged-by models
        (besides "otu" and "project").  The last component must be a field
        name.  'natural' is allowed.  For non-averaged data, the default, i.e.
        when an empty list/tuple is given, is equivalent to using
        'sequencing__natural' for non-averaged data.  For averaged data, the
        averaged-by columns are always added and meta_cols is used for extra
        columns.

        :param int min_avg_group_size:
            Minimum group size when averaging.  This should be 1, the default,
            or larger.  When it is larger than 1, then rows resulting from
            averaging values less then the given number of samples are skipped.

        Returns an iterator over tuple rows, first row is the header.  This is
        intended to support data export.  Missing counts are inserted as zeros,
        complementing the skipping of zeros at import.

        Will raise an exception when data come from multiple projects.
        """
        log.debug(f'as_shared_values_list() args: normalize={normalize} '
                  f'meta_cols={meta_cols}')
        if mothur:
            raise ValueError('Mothur mode is not implement')

        # ensure we have a project before continuing
        if self._project is None:
            # filter_project() raises on data from multiple projects
            return self.filter_project().as_shared_values_list(
                normalize=normalize,
                meta_cols=meta_cols,
                mothur=mothur,
            )

        if self._avg_by:
            # id_fields are the avg_by fields without project and otu
            # they will be the row identifiers:
            id_fields = [
                i for i in self._avg_by
                if i not in ['project', 'otu']
            ]
            meta_models = [
                Abundance.get_field(i).related_model
                for i in id_fields
            ]
            # Default accessors are derived from the id fields. For simplicity
            # these get hard-coded here to go first in the meta-cols in the
            # same order as id_fields
            # TODO: make adding the default in caller-supplied meta_col matter
            # somehow
            defaults = [
                i._meta.model_name + '__natural'
                for i in meta_models
            ]
            meta_cols = defaults + [i for i in meta_cols if i not in defaults]
        else:
            id_fields = ['sequencing']
            meta_models = [Sequencing]
            # setting default here
            if not meta_cols:
                meta_cols = ['sequencing__name']

        meta_cols1 = []
        meta_cols_verbose = []
        for a in meta_cols:
            a = a.split('__')
            for model in meta_models:
                if a[0] == model._meta.model_name:
                    break
            else:
                raise ValueError(
                    f'Meta column accessor {a} does not match any meta field '
                    f'model in {meta_models}'
                )
            meta_cols1.append((model, '__'.join(a[1:])))
            verb, last = a[-2:]
            verb = verb.capitalize()
            if last not in ('name', 'natural', 'id', 'pk'):
                verb = verb + '.' + last
            meta_cols_verbose.append(verb)
        meta_cols = meta_cols1
        del meta_cols1, verb, last, a

        # get OTUs that actually occur in QuerySet:
        otu_pks = set(self.values_list('otu', flat=True).distinct().iterator())
        # OTU order here must correspond to order in which count values are
        # generated later, in the _group_and_pivot() method.  It is assumed
        # that the OTU model defines an appropriate order; OTU pk->name
        # mapping, use values for header, keys for zeros injection
        otus = OrderedDict((
            (i.pk, i.natural)
            for i in OTU.objects.iterator()
            if i.pk in otu_pks
        ))

        # Build header row:
        if mothur:
            header = ['label', 'Group', 'numOtus']
        else:
            header = list(meta_cols_verbose)
        if self._avg_by:
            header.append('avg_group_count')
        header += list(otus.values())

        log.debug(f'shared export prep: id_fields={id_fields} meta={meta_cols}'
                  f' verbose={meta_cols_verbose}')

        tail = self._group_and_pivot(
            list(otus.keys()),
            normalize,
            id_fields,
            min_avg_group_size,
        )
        tail = self._add_meta_data(tail, meta_cols)
        return chain([header], tail)

    def as_shared_dataframe(
            self,
            normalize=0,
            meta_cols=(),
    ):
        """
        Get "shared" table as pandas data frame

        This is a wrapper around as_shared_values_list()

        Will raise an exception when data come from multiple projects.
        """
        it = self.as_shared_values_list(
            normalize=normalize,
            meta_col=meta_cols
        )
        cols = next(it)
        return DataFrame(it, columns=cols)


class Abundance(Model):
    """
    Models read count

    *** HOWTO add data from an analysis run ***

    (1) create a new AnalysisProject record.
    (2) For a mothur SOP 97% OTU run, abundances only, no sequences, you need
        these files:

        ugrads19.asv0.precluster.pick.opti_mcc.shared
        ugrads19.asv0.precluster.pick.opti_mcc.0.03.rep.fasta
        ugrads19.asv0.precluster.pick.opti_mcc.0.03.cons.taxonomy

    (3) Then in the shell, run:

      1 from mibios_seq.models import Abundance, AnalysisProject
      2 p = AnalysisProject.objects.get(name='mothur_SOP_97pct')
      3 Abundance.from_file(
            'ugrads19.final.shared',
            project=p,
            fasta='ugrads19.final.fa'
        )
    """
    history = None
    otu = models.ForeignKey(
        'OTU',
        on_delete=models.CASCADE,
        editable=False,
        verbose_name='OTU',
    )
    count = models.PositiveIntegerField(
        help_text='absolute abundance',
        editable=False,
    )
    relative = models.FloatField(
        default=None, null=True,
        verbose_name='relative abundance',
        editable=False,
    )
    sequencing = models.ForeignKey(
        Sequencing,
        on_delete=models.CASCADE,
        editable=False,
    )
    project = models.ForeignKey(
        'AnalysisProject',
        on_delete=models.CASCADE,
        editable=False,
        verbose_name='analysis project',
    )

    class Meta:
        unique_together = (
            # one count per project / OTU / sample
            ('otu', 'sequencing', 'project'),
        )
        verbose_name_plural = 'abundance'

    objects = Manager.from_queryset(AbundanceQuerySet)()
    curated = CurationManager.from_queryset(AbundanceQuerySet)()

    average_by = [('project', 'otu',
                   'sequencing__sample__fecalsample__participant',
                   'sequencing__sample__fecalsample__week')]
    average_fields = ['relative']

    def __str__(self):
        return super().__str__() + f' |{self.count}|'

    @classmethod
    def from_file(cls, file, project, fasta=None, threads=1):
        """
        Load abundance data from shared file

        :param file fasta: Fasta file object
        :param str otu_type: A valid OTU type.

        If a fasta file is given, then the input does not need to use proper
        ASV numbers.  Instead ASVs are identified by sequence and ASV objects
        are created as needed.  Obviously, the OTU/ASV/sequence names in shared
        and fasta files must correspond.
        """
        sh = MothurShared(file, verbose=False, threads=threads)
        return cls._from_file(file, project, fasta, sh)

    @classmethod
    @atomic
    def _from_file(cls, file, project, fasta, sh):
        if fasta:
            fasta_result = OTU.from_fasta(fasta, project=project)
        else:
            fasta_result = None
        AbundanceImportFile.create_from_file(file=file, project=project)
        sequencings = Sequencing.objects.in_bulk(field_name='name')
        if project.otu_type == AnalysisProject.ASV_TYPE:
            f = dict(project=None)
        else:
            f = dict(project=project)
        otus = {
            (i.prefix, i.number): i
            for i in OTU.objects.all().filter(**f).iterator()
        }
        del f

        skipped, zeros, otus_new = 0, 0, 0
        objs = []
        for (seqid, otu), count in sh.counts.stack().items():
            if count == 0:
                # don't store zeros
                zeros += 1
                continue

            if seqid not in sequencings:
                # ok to skip, e.g. non-public
                skipped += 1
                continue

            try:
                otu_key = OTU.natural_lookup(otu)
            except ValueError:
                raise UserDataError(
                    f'Irregular OTU identifier not supported: {otu}'
                )
            else:
                otu_key = (otu_key['prefix'], otu_key['number'])

            try:
                otu_obj = otus[otu_key]
            except KeyError:
                otu_obj = OTU.objects.create(
                    prefix=otu_key[0],
                    number=otu_key[1],
                    project=project,
                )
                otus[otu_key] = otu_obj
                otus_new += 1

            objs.append(cls(
                count=count,
                project=project,
                sequencing=sequencings[seqid],
                otu=otu_obj,
            ))

        cls.objects.bulk_create(objs)
        return dict(count=len(objs), zeros=zeros, skipped=skipped,
                    fasta=fasta_result, otus_created=otus_new)

    @classmethod
    def compute_relative(cls, project=None):
        """
        Compute values for the relative abundance field

        :param project: Restrict calculations to given project

        This will overwrite existing data.  If case of errors, partial updates
        are possible.
        """
        qs = (cls.objects
              .select_related('project', 'sequencing')
              .order_by('project__pk', 'sequencing__pk'))

        if project:
            qs = qs.filter(project=project)

        grpkey = attrgetter('project.pk', 'sequencing.pk')
        for _, group in groupby(qs.iterator(), key=grpkey):
            group = list(group)
            total = sum((i.count for i in group))
            for i in group:
                i.relative = i.count / total
            cls.objects.bulk_update(group, ('relative', ))

    @classmethod
    def compare_projects(cls, project_a, project_b):
        """
        Compare absolute counts between two ASV analysis projects
        """
        if project_a == project_b:
            raise ValueError('the two given projects must not be the same')

        seqs_a = project_a.sequencing.distinct().values_list('pk', flat=True)
        seqs_a = set(seqs_a.iterator())
        seqs_b = project_b.sequencing.distinct().values_list('pk', flat=True)
        seqs_b = set(seqs_b.iterator())

        # rows are 4-tuples of int with 3 pks + count
        qs = cls.objects.filter(project__in=[project_a, project_b])
        qs = qs.order_by('sequencing', 'otu', 'project')
        qs = qs.values_list('sequencing', 'otu', 'project', 'count')

        def keyfun(row):
            # group by sequencing and otu
            return (row[0], row[1])

        total = 0
        skipped = 0
        same = 0
        diffs = {}
        for _, grp in groupby(qs.iterator(), key=keyfun):
            total += 1
            (seq, otu, proj, count), *more = grp
            if more:
                if len(more) > 1:
                    raise RuntimeError('expected at most two in group')
                # more has one element
                count_b = more[0][3]
                delta = count_b - count
                max_count = max(count, count_b)
            else:
                if proj == project_a.pk and seq in seqs_b:
                    # zero in project b
                    delta = -count
                elif proj == project_b.pk and seq in seqs_a:
                    # zero in project a
                    delta = count
                else:
                    # sequencing data was not analysed in other project
                    skipped += 1
                    continue
                max_count = count

            if delta == 0:
                same += 1
                continue

            pct = abs(delta) / max_count
            diffs[(seq, otu)] = (delta, pct)

        return dict(
            total=total,
            skipped=skipped,
            same=same,
            diffs=diffs,
        )


class AnalysisProject(Model):
    ASV_TYPE = 'ASV'
    PCT97_TYPE = '97pct'
    OTU_TYPE_CHOICES = (
        (ASV_TYPE, 'ASV'),
        (PCT97_TYPE, '97% OTU'),
    )

    name = models.CharField(max_length=100, unique=True)
    otu = models.ManyToManyField('OTU', through=Abundance, editable=False)
    sequencing = models.ManyToManyField(Sequencing, through=Abundance,
                                        editable=False, related_name='project')
    otu_type = models.CharField(max_length=5, choices=OTU_TYPE_CHOICES,
                                verbose_name='OTU type')
    description = models.TextField(blank=True)

    @classmethod
    def get_fields(cls, with_m2m=False, **kwargs):
        # Prevent numbers from being displayed, too much data
        return super().get_fields(with_m2m=False, **kwargs)


class OTUQuerySet(QuerySet):
    def to_fasta(self, save_as=None):
        """
        Convert OTU queryset into a fasta file
        """
        if not save_as:
            return self._to_fasta()

        with open(save_as, 'w') as f:
            for i in self._to_fasta():
                f.write(i)

    def _to_fasta(self):
        qs = self.select_related('sequence').iterator()
        for i in qs:
            yield f'>{i.natural}\n{i.sequence.seq}\n'


class OTU(Model):
    NUM_WIDTH = 5

    prefix = models.CharField(max_length=8)
    number = models.PositiveIntegerField()
    project = models.ForeignKey('AnalysisProject', on_delete=models.CASCADE,
                                related_name='owned_otus',
                                null=True, blank=True)
    sequence = models.ForeignKey(
        Sequence,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        editable=False,
    )

    hidden_fields = ['prefix', 'number']  # use name property instead

    objects = Manager.from_queryset(OTUQuerySet)()
    curated = CurationManager.from_queryset(OTUQuerySet)()

    class Meta:
        ordering = ('prefix', 'number',)
        unique_together = (('prefix', 'number', 'project'),)
        verbose_name = 'OTU'

    def __str__(self):
        return str(self.natural)
        s = str(self.natural)
        if self.taxon:
            genus, _, species = self.taxon.name.partition(' ')
            if species:
                genus = genus.lstrip('[')[0].upper() + '.'
                s += ' ' + genus + ' ' + species
            else:
                s += ' ' + str(self.taxon)

        return s

    @property
    def name(self):
        return str(self.natural)

    @Model.natural.getter
    def natural(self):
        return self.prefix + '{}'.format(self.number).zfill(self.NUM_WIDTH)

    @classmethod
    def natural_lookup(cls, value):
        """
        Given e.g. ASV00023, return dict(number=23)

        Raises ValueError if value does not end with a number.
        Raises KeyError if value has no non-numeric prefix.
        """
        if value is cls.NOT_A_VALUE:
            return dict(prefix=None, number=None)

        places = 0
        while value[-1 - places].isdecimal():
            places += 1

        number = int(value[-places:])
        prefix = value[:-places]
        return dict(prefix=prefix, number=number)

    @classmethod
    @atomic
    def from_fasta(cls, file, project=None):
        """
        Import from given fasta file

        :param file: Fasta-formatted input file
        :param AnalysisProject project: AnalysisProject that generated the OTUs
                                        If this is None, then the OTU type will
                                        be set to ASV.
        """
        file_rec = ImportFile.create_from_file(file=file)

        try:
            file_rec.file.open('r')
            return cls._from_fasta(file_rec, project)
        except Exception:
            try:
                file_rec.file.close()
                file_rec.file.delete()
            except Exception:
                pass
            raise
        else:
            file_rec.file.close()

    @classmethod
    def _from_fasta(cls, file_rec, project):
        added, updated, skipped, total = 0, 0, 0, 0
        seq_added = 0

        # passing the filehandle to SeqIO.parse: The SeqIO fasta parser tries
        # to skip over comments and empty lines at the begin of the file by
        # iterating over the passed file handle.  After the first line with '>'
        # is found, the line is kept and then it breaks out of the for loop and
        # enters another for loop over the file handle iterator to get the rest
        # of the data.  When we pass a django.core.files.base.File object the
        # second loop entry calls a seek(0) as part of the chunking machinery
        # and it gets messy.  This is why we pass the underlying file object
        # and hope this won't blow up when something about the file storage
        # changes.
        for i in SeqIO.parse(file_rec.file.file.file, 'fasta'):
            try:  # expect {'prefix': X, 'number': N}
                kwnum = cls.natural_lookup(i.id)
            except ValueError:
                # SeqIO sequence id does not parse,
                # is something from analysis pipeline, no OTU number?
                skipped += 1
                continue

            seq, new_seq = Sequence.objects.get_or_create(seq=i.seq)
            if new_seq:
                seq_added += 1

            if project and project.otu_type == project.ASV_TYPE:
                # ASVs do not belong to a project
                project = None

            obj_kw = dict(
                prefix=kwnum['prefix'],
                number=kwnum['number'],
                project=project,
            )
            has_changed = False
            try:
                obj = cls.objects.get(**obj_kw)
            except cls.DoesNotExist:
                obj = cls(sequence=seq, **obj_kw)
                added += 1
                has_changed = True
            else:
                if obj.sequence is None:
                    obj.sequence = seq
                    updated += 1
                    has_changed = True
                elif obj.sequence != seq:
                    raise UserDataError(f'OTU record already exists with'
                                        f'different sequence: {obj}')

            if has_changed:
                try:
                    obj.full_clean()
                except Exception as e:
                    log.error('Failed importing ASV: at fasta record '
                              f'{total + 1}: {i}: {e}')
                    raise

                obj.add_change_record(file=file_rec, line=total + 1)
                obj.save()

            total += 1

        return dict(total=total, new=added, updated=updated,
                    skipped=skipped)


class Taxonomy(Model):
    taxid = models.PositiveIntegerField(
        unique=True,
        verbose_name='NCBI taxonomy id',
    )
    name = models.CharField(
        max_length=300,
        unique=True,
        verbose_name='taxonomic name',
    )

    class Meta:
        verbose_name_plural = 'taxonomy'

    def __str__(self):
        return '{} ({})'.format(self.name, self.taxid)

    @classmethod
    @atomic
    def from_blast_top1(cls, file):
        """
        Import from blast-result-top-1 format file

        The supported file format is a tab-delimited text file with header row,
        column 1 are ASV accessions, columns 5 and 6 are NCBI taxids and names,
        and if there are ties then column 7 are the least-common NCBI taxids
        and column 8 are the corresponding taxon names

        The taxonomy for existing ASV records is imported, everything else is
        ignored.
        """
        file_rec = ImportFile.create_from_file(file=file)
        otus = {(i.prefix, i.number): i for i in OTU.objects.select_related()}
        is_header = True  # first line is header
        updated, total, seq_missing = 0, 0, 0
        seqs = []
        for line in file_rec.file.open('r'):
            if is_header:
                is_header = False
                continue

            try:
                total += 1
                row = line.rstrip('\n').split('\t')
                asv, taxid, name, lctaxid, lcname = row[0], *row[4:]

                if lcname and lcname:
                    name = lcname
                    taxid = lctaxid

                taxid = int(taxid)
                match = OTU.natural_lookup(asv)
                prefix = match['prefix']
                num = match['number']
                del match

                if (prefix, num) not in otus:
                    # ASV not in database
                    continue

                try:
                    taxon = cls.objects.get(taxid=taxid, name=name)
                except cls.DoesNotExist:
                    taxon = cls(taxid=taxid, name=name)
                    taxon.full_clean()
                    taxon.add_change_record(file=file_rec, line=total + 1)
                    taxon.save()

                seq = otus[(prefix, num)].sequence
                if seq is None:
                    seq_missing += 1
                else:
                    if seq.taxon != taxon:
                        seq.taxon = taxon
                        updated += 1
                seqs.append(seq)
            except Exception as e:
                raise RuntimeError(
                    f'error loading file: {file} at line {total}: {row}'
                ) from e

        Sequence.objects.bulk_update(seqs, ['taxon'])
        return dict(total=total, update=updated, seq_missing=seq_missing)


class AbundanceImportFile(ImportFile):
    """
    An import file that keeps tab to which project it belongs

    Since Abundance opts out of history this connecting the import file with
    the project keeps at leas some record of the origin of abundance data.
    """
    project = models.ForeignKey(AnalysisProject, on_delete=models.CASCADE)
