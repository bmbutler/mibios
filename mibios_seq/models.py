from collections import OrderedDict
from itertools import chain, groupby
from operator import itemgetter

from Bio import SeqIO
from django.db import models
from django.db.transaction import atomic

from omics.shared import MothurShared
from mibios.dataset import UserDataError
from mibios.models import (ImportFile, Manager, PublishManager, Model,
                           ParentModel, QuerySet)
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


class SeqNote(Model):
    name = models.CharField(max_length=100, unique=True)
    text = models.TextField(max_length=5000, blank=True)


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
    sample = models.ForeignKey(Sample, on_delete=models.CASCADE,
                               blank=True, null=True)
    control = models.CharField(max_length=50, choices=CONTROL_CHOICES,
                               blank=True)
    r1_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    r2_file = models.CharField(max_length=300, unique=True, blank=True,
                               null=True)
    note = models.ManyToManyField(SeqNote, blank=True)
    run = models.ForeignKey('SequencingRun', on_delete=models.CASCADE,
                            blank=True, null=True)
    plate = models.PositiveSmallIntegerField(blank=True, null=True)
    plate_position = models.CharField(max_length=10, blank=True)
    snumber = models.PositiveSmallIntegerField(blank=True, null=True)

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
        s, n = value.split('-')
        return dict(serial=s, number=int(n))


class Strain(Model):
    asv = models.ForeignKey('ASV', on_delete=models.SET_NULL, blank=True,
                            null=True)


class AbundanceQuerySet(QuerySet):
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
            .as_dataframe('asv', 'sequencing', 'count', natural=True)
            .pivot(index='sequencing', columns='asv', values='count')
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
        Scale counts to normal sample size
        """
        group = list(group)
        debug <= 1 or print(f'{group=}')
        vals = [i[0] for i in group]
        debug <= 1 or print(f'{vals=}')
        total = sum(vals)
        debug <= 0 or print(f'{total=}')
        f = size / total
        frac = [i * f for i in vals]
        debug <= 1 or print(f'{frac=}')
        disc = [round(i) for i in frac]
        overhang = sum(disc) - size
        debug <= 0 or print(f'{overhang=}')
        for _ in range(abs(overhang)):
            errs = [i - j for i, j in zip(disc, frac)]  # round-up errs are pos
            debug <= 1 or print(f'{disc=} {sum(disc)=}')
            debug <= 1 or print(f'{errs=}')
            if overhang > 0:
                abs_err_max = max(errs)
            else:
                abs_err_max = min(errs)
            idx = errs.index(abs_err_max)
            if overhang > 0:
                disc[idx] -= 1
            else:
                disc[idx] += 1
            debug <= 0 or print(f'{idx=} {disc[idx]=} err: '
                                '{errs[idx]}->{disc[idx]-frac[idx]}')
        debug <= 1 or print(f'{disc=} {sum(disc)=}')
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
        """
        group = list(group)
        vals = [i[0] for i in group]
        total = sum(vals)
        return ((i / total, j, k) for i, j, k in group)

    def _shared_file_items(self, asvs, group):
        """
        Generator over the items of a row in a shared file

        :param asvs: iterable over all ASV pks of QuerySet in order
        :param group: Data for one sample, row; an iterable over triplets
                      (count, seq name, ASV pk)
        """
        count, _, asv_pk = next(group)
        for i in asvs:
            if asv_pk == i:
                yield count
                try:
                    count, _, asv_pk = next(group)
                except StopIteration:
                    pass
            else:
                yield 0

    def _shared_file_rows(self, asvs, normalize=None):
        """
        Generator of shared file rows

        :param int normalize: Normalization mode.  If None then absolute counts
        are returned.  If 0 then relative abundance, as fractional values in
        [0, 1] are returned.  If an integer above 0 is given, then the counts
        are normalized by 'discrete scaling' to the targeted sample size.
        """
        it = (
            self.order_by('sequencing', 'asv')
            .values_list('count', 'sequencing__name', 'asv')
            .iterator()
        )
        for name, group in groupby(it, key=itemgetter(1)):
            if normalize == 0:
                group = self._unit_normalize(group)
            elif normalize is not None:
                group = self._normalize(group, size=normalize)
            yield tuple(chain([name], self._shared_file_items(asvs, group)))

    def as_shared_values_list(self, normalize=None):
        """
        Make mothur shared table for download

        Returns an iterator over tuple rows, first row is the header.  This is
        intended to support data export.  Missing counts are inserted as zero,
        mirroring the skipping of zeros at import.
        """
        # get ASVs that actually occur in QuerySet:
        asv_pks = set(self.values_list('asv', flat=True).distinct().iterator())
        # ASV order here must correspond to order in which count values are
        # generated later, in the _shared_file_rows() method.  It is assumed
        # that the ASV model defines an appropriate order; ASV pk->name
        # mapping, use values for header, keys for zeros injection
        asvs = OrderedDict((
            (i.pk, i.natural)
            for i in ASV.objects.iterator()
            if i.pk in asv_pks
        ))
        header = ['Group'] + list(asvs.values())

        return chain([header], self._shared_file_rows(
            list(asvs.keys()),
            normalize=normalize,
        ))


class Abundance(Model):
    history = None
    name = models.CharField(
        max_length=50, verbose_name='project internal id',
        default='',
        blank=True,
        help_text='project specific ASV/OTU identifier',
    )
    count = models.PositiveIntegerField(
        help_text='absolute abundance',
        editable=False,
    )
    project = models.ForeignKey(
        'AnalysisProject',
        on_delete=models.CASCADE,
        editable=False,
    )
    sequencing = models.ForeignKey(
        Sequencing,
        on_delete=models.CASCADE,
        editable=False,
    )
    asv = models.ForeignKey(
        'ASV',
        on_delete=models.CASCADE,
        editable=False,
    )

    class Meta:
        unique_together = (
            # one count per project / ASV / sample
            ('name', 'sequencing', 'project'),
            ('asv', 'sequencing', 'project'),
        )

    objects = Manager.from_queryset(AbundanceQuerySet)()
    published = PublishManager.from_queryset(AbundanceQuerySet)()

    def __str__(self):
        return super().__str__() + f' |{self.count}|'

    @classmethod
    def from_file(cls, file, project, fasta=None):
        """
        Load abundance data from shared file

        :param file fasta: Fasta file object

        If a fasta file is given, then the input does not need to use proper
        ASV numbers.  Instead ASVs are identified by sequence and ASV objects
        are created as needed.  Obviously, the OTU/ASV/sequence names in shared
        and fasta files must correspond.
        """
        sh = MothurShared(file, verbose=False, threads=1)
        with atomic():
            if fasta:
                fasta_result = ASV.from_fasta(fasta)

            AbundanceImportFile.create_from_file(file=file, project=project)
            sequencings = Sequencing.published.in_bulk(field_name='name')
            asvs = ASV.published.in_bulk(field_name='number')  # get numbered

            if fasta:
                asvs.update(fasta_result['irregular'])

            skipped, zeros = 0, 0
            objs = []
            for (sample, asv), count in sh.counts.stack().items():
                if count == 0:
                    # don't store zeros
                    zeros += 1
                    continue

                if sample not in sequencings:
                    # ok to skip, e.g. non-public
                    skipped += 1
                    continue

                try:
                    asv_key = ASV.natural_lookup(asv)['number']
                except ValueError:
                    asv_key = asv

                try:
                    asv_obj = asvs[asv_key]
                except KeyError:
                    raise UserDataError(f'Unknown ASV: {asv}')

                objs.append(cls(
                    name=asv,
                    count=count,
                    project=project,
                    sequencing=sequencings[sample],
                    asv=asv_obj,
                ))

            cls.published.bulk_create(objs)
        return dict(count=len(objs), zeros=zeros, skipped=skipped)


class AnalysisProject(Model):
    name = models.CharField(max_length=100, unique=True)
    asv = models.ManyToManyField('ASV', through=Abundance, editable=False)
    description = models.TextField(blank=True)

    @classmethod
    def get_fields(cls, with_m2m=False, **kwargs):
        # Prevent abundance from being displayed, too much data
        return super().get_fields(with_m2m=False, **kwargs)


class ASV(Model):
    PREFIX = 'ASV'
    NUM_WIDTH = 5

    number = models.PositiveIntegerField(null=True, blank=True, unique=True)
    taxon = models.ForeignKey('Taxonomy', on_delete=models.SET_NULL,
                              blank=True, null=True)
    sequence = models.CharField(
        max_length=300,  # > length of 16S V4
        unique=True,
        editable=False,
    )

    class Meta:
        ordering = ('number',)

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
        if self.number:
            return self.PREFIX + '{}'.format(self.number).zfill(self.NUM_WIDTH)
        else:
            return self.pk

    @classmethod
    def natural_lookup(cls, value):
        """
        Given e.g. ASV00023, return dict(number=23)

        Raises ValueError if value does not parse
        """
        # FIXME: require casefolded ASV prefix?
        return dict(number=int(value[len(cls.PREFIX):]))

    @classmethod
    @atomic
    def from_fasta(cls, file):
        """
        Import from given fasta file

        For fasta headers that do not have ASV00000 type id the returned
        'irregular' dict will map the irregular names to the corresponding ASV
        instance.  Re-loading un-numbered sequences with a proper ASV number
        will get the number updated.  If a sequence already has a number and it
        doesn't match the number in the file an IntegrityError is raised.
        """
        file_rec = ImportFile.create_from_file(file=file)

        try:
            file_rec.file.open('r')
            return cls._from_fasta(file_rec)
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
    def _from_fasta(cls, file_rec):
        added, updated, total = 0, 0, 0
        irregular = {}

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
            try:
                kwnum = cls.natural_lookup(i.id)  # expect {'number': N}
            except ValueError:
                # SeqIO sequence id does not parse,
                # is something from analysis pipeline, no ASV number
                kwnum = {}

            try:
                obj = cls.objects.get(sequence=i.seq)
            except cls.DoesNotExist:
                obj = cls(sequence=i.seq, **kwnum)
                added += 1
            else:
                if obj.number is None and kwnum:
                    # update number
                    for k, v in kwnum.items():
                        setattr(obj, k, v)
                    updated += 1

            if obj.pk is None or updated:
                try:
                    obj.full_clean()
                except Exception as e:
                    log.error('Failed importing ASV: at fasta record '
                              f'{total + 1}: {i}: {e}')
                    raise

                obj.add_change_record(file=file_rec, line=total + 1)
                obj.save()

            if not kwnum:
                # save map from irregular sequence id to ASV
                irregular[i.id] = obj

            total += 1

        return dict(total=total, new=added, updated=updated,
                    irregular=irregular)


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
        asvs = {i.number: i for i in ASV.objects.select_related()}
        is_header = True  # first line is header
        updated, total = 0, 0
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
                num = ASV.natural_lookup(asv)['number']

                if num not in asvs:
                    # ASV not in database
                    continue

                try:
                    taxon = cls.objects.get(taxid=taxid, name=name)
                except cls.DoesNotExist:
                    taxon = cls(taxid=taxid, name=name)
                    taxon.full_clean()
                    taxon.add_change_record(file=file_rec, line=total + 1)
                    taxon.save()

                if asvs[num].taxon == taxon:
                    del asvs[num]
                else:
                    asvs[num].taxon = taxon
                    updated += 1
            except Exception as e:
                raise RuntimeError(
                    f'error loading file: {file} at line {total}: {row}'
                ) from e

        ASV.objects.bulk_update(asvs.values(), ['taxon'])
        return dict(total=total, update=updated)


class AbundanceImportFile(ImportFile):
    """
    An import file that keeps tab to which project it belongs

    Since Abundance opts out of history this connecting the import file with
    the project keeps at leas some record of the origin of abundance data.
    """
    project = models.ForeignKey(AnalysisProject, on_delete=models.CASCADE)
