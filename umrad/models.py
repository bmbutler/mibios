from logging import getLogger
from itertools import chain
import os
import sys

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.transaction import atomic

from mibios.models import Model

from .fields import AccessionField, DataPathField


log = getLogger(__name__)

# standard data field options
opt = dict(blank=True, null=True, default=None)  # non-char optional
ch_opt = dict(blank=True, default='')  # optional char
uniq_opt = dict(unique=True, **opt)  # unique and optional (char/non-char)
# standard foreign key options
fk_req = dict(on_delete=models.CASCADE)  # required FK
fk_opt = dict(on_delete=models.SET_NULL, **opt)  # optional FK


class EOF():
    """
    End-of-file marker for file iteration
    """
    def __len__(self):
        return 0


class Bin(Model):
    history = None
    sample = models.ForeignKey('Sample', **fk_req)
    number = models.PositiveIntegerField()
    checkm = models.OneToOneField('CheckM', **fk_opt)

    method = None  # set by inheriting model

    class Meta:
        abstract = True
        unique_together = (
            ('sample', 'number'),
        )

    def __str__(self):
        return f'{self.sample.accession} {self.method} #{self.number}'

    @classmethod
    def get_class(cls, method):
        if cls.method is not None:
            return super().get_class(method)

        for i in cls.__subclasses__():
            if i.method == method:
                return i

        raise ValueError(f'not a valid binning type/method: {method}')

    @classmethod
    def import_all(cls):
        """
        Import all binning data

        This class method can be called on the abstract parent Bin class and
        will then import data for all binning types.  Or it can be called on an
        inheriting class and then will only import data for the corresponding
        binning type.
        """
        if not cls._meta.abstract:
            raise RuntimeError(
                'method can not be called by concrete bin subclass'
            )
        for i in Sample.objects.all():
            cls.import_sample_bins(i)

    @classmethod
    def import_sample_bins(cls, sample):
        """
        Import bin for given sample
        """
        if sample.binning_ok:
            log.info(f'{sample} has bins loaded already')
            return

        if cls._meta.abstract:
            # Bin parent class only
            with atomic():
                noerr = True
                for klass in cls.__subclasses__():
                    res = klass.import_sample_bins(sample)
                    noerr = bool(res) and noerr
                if noerr:
                    sample.binning_ok = True
                    sample.save()
                return

        # Bin subclasses only
        files = list(cls.bin_files(sample))
        if not files:
            log.warning(f'no {cls.method} bins found for {sample}')
            return None

        for i in sorted(files):
            res = cls._import_bins_file(sample, i)
            if not res:
                log.warning(f'got empty cluster from {i} ??')

        return len(files)

    @classmethod
    def _import_bins_file(cls, sample, path):
        _, num, _ = path.name.split('.')
        try:
            num = int(num)
        except ValueError as e:
            raise RuntimeError(f'Failed parsing filename {path}: {e}')

        obj = cls.objects.create(sample=sample, number=num)
        cids = []
        with path.open() as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                cids.append(line.strip().lstrip('>'))

        qs = ContigCluster.objects.filter(sample=sample, cluster_id__in=cids)
        kwargs = {cls._meta.get_field('members').remote_field.name: obj}
        qs.update(**kwargs)
        log.info(f'{obj} imported: {len(cids)} contig clusters')
        return len(cids)

    @classmethod
    def bin_files(cls, sample):
        """
        Generator over bin file paths
        """
        pat = f'{sample.accession}_{cls.method}_bins.*'
        path = settings.GLAMR_DATA_ROOT / 'BINS'
        return path.glob(pat)


class BinMAX(Bin):
    method = 'MAX'


class BinMET93(Bin):
    method = 'MET_P97S93E300'


class BinMET97(Bin):
    method = 'MET_P99S97E300'


class BinMET99(Bin):
    method = 'MET_P99S99E300'


class CheckM(Model):
    """
    CheckM results for a bin
    """
    history = None
    translation_table = models.PositiveSmallIntegerField(
        verbose_name='Translation table',
    )
    gc_std = models.FloatField(verbose_name='GC std')
    ambiguous_bases = models.PositiveIntegerField(
        verbose_name='# ambiguous bases',
    )
    genome_size = models.PositiveIntegerField(verbose_name='Genome size')
    longest_contig = models.PositiveIntegerField(verbose_name='Longest contig')
    n50_scaffolds = models.PositiveIntegerField(verbose_name='N50 (scaffolds)')
    mean_scaffold_len = models.FloatField(verbose_name='Mean scaffold length')
    num_contigs = models.PositiveIntegerField(verbose_name='# contigs')
    num_scaffolds = models.PositiveIntegerField(verbose_name='# scaffolds')
    num_predicted_genes = models.PositiveIntegerField(
        verbose_name='# predicted genes',
    )
    longest_scaffold = models.PositiveIntegerField(
        verbose_name='Longest scaffold',
    )
    gc = models.FloatField(verbose_name='GC')
    n50_contigs = models.PositiveIntegerField(verbose_name='N50 (contigs)')
    coding_density = models.FloatField(verbose_name='Coding density')
    mean_contig_length = models.FloatField(verbose_name='Mean contig length')

    @classmethod
    def import_all(cls):
        for i in Sample.objects.all():
            if i.checkm_ok:
                log.info(f'sample {i}: have checkm stats, skipping')
                continue
            cls.import_sample(i)

    @classmethod
    @atomic
    def import_sample(cls, sample):
        bins = {}
        stats_file = sample.get_checkm_stats_path()
        if not stats_file.exists():
            log.warning(f'{sample}: checkm stats do not exist: {stats_file}')
            return

        for binid, obj in cls.from_file(stats_file):
            # parse binid, is like: Sample_42895_MET_P99S99E300_bins.6
            part1, _, num = binid.partition('.')  # separate number part
            parts = part1.split('_')
            sample_name = '_'.join(parts[:2])
            meth = '_'.join(parts[2:-1])  # rest but without the "_bins"

            try:
                num = int(num)
            except ValueError as e:
                raise ValueError(f'Bad bin id in stats: {binid}, {e}')

            if sample_name != sample.accession:
                raise ValueError(
                    f'Bad sample name in stats: {binid} -- expected: '
                    f'{sample.accession}'
                )

            try:
                binclass = Bin.get_class(meth)
            except ValueError as e:
                raise ValueError('Bad method in stats: {binid}: {e}') from e

            try:
                binobj = binclass.objects.get(sample=sample, number=num)
            except binclass.DoesNotExist as e:
                raise RuntimeError(
                    f'no bin with checkm bin id: {binid} file: {stats_file}'
                ) from e

            binobj.checkm = obj

            if binclass not in bins:
                bins[binclass] = []

            bins[binclass].append(binobj)

        for binclass, bobjs in bins.items():
            binclass.objects.bulk_update(bobjs, ['checkm'])

        sample.checkm_ok = True
        sample.save()

    @classmethod
    def from_file(cls, path):
        """
        Create instances from given bin_stats.analyze.tsv file.

        Should normally be only called from CheckM.import_sample().
        """
        ret = []
        with path.open() as fh:
            for line in fh:
                bin_key, data = line.strip().split('\t')
                data = data.lstrip('{').rstrip('}').split(', ')
                obj = cls()
                for item in data:
                    key, value = item.split(': ', maxsplit=2)
                    key = key.strip("'")
                    for i in cls._meta.get_fields():
                        if i.is_relation:
                            continue
                        # assumes correclty assigned verbose names
                        if i.verbose_name == key:
                            field = i
                            break
                    else:
                        raise RuntimeError(
                            f'Failed parsing {path}: no matching field for '
                            f'{key}, offending line is:\n{line}'
                        )

                    try:
                        value = field.to_python(value)
                    except ValidationError as e:
                        message = (f'Failed parsing field "{key}" on line:\n'
                                   f'{line}\n{e.message}')
                        raise ValidationError(
                            message,
                            code=e.code,
                            params=e.params,
                        ) from e

                    setattr(obj, field.attname, value)

                obj.save()
                ret.append((bin_key, obj))

        return ret


class ContigCluster(Model):
    history = None
    cluster_id = AccessionField(prefix='CLUSTER', unique=False, max_length=50)
    sample = models.ForeignKey('Sample', **fk_req)

    # Data from assembly:
    # 1. offset of begin of sequence in bytes
    seq_offset = models.PositiveIntegerField(**opt)
    # 2. num of bytes until next offset (or EOF)
    seq_bytes = models.PositiveIntegerField(**opt)

    # Data from mapping / coverage:
    # decimals in bbmap output have 4 fractional places
    # FIXME: determine max_places for sure
    length = models.PositiveIntegerField(**opt)
    bases = models.PositiveIntegerField(**opt)
    coverage = models.DecimalField(decimal_places=4, max_digits=10, **opt)
    reads_mapped = models.PositiveIntegerField(**opt)
    rpkm = models.DecimalField(decimal_places=4, max_digits=10, **opt)
    frags_mapped = models.PositiveIntegerField(**opt)
    fpkm = models.DecimalField(decimal_places=4, max_digits=10, **opt)

    # bin membership
    bin_max = models.ForeignKey(BinMAX, **fk_opt, related_name='members')
    bin_m93 = models.ForeignKey(BinMET93, **fk_opt, related_name='members')
    bin_m97 = models.ForeignKey(BinMET97, **fk_opt, related_name='members')
    bin_m99 = models.ForeignKey(BinMET99, **fk_opt, related_name='members')

    class Meta:
        unique_together = (
            ('sample', 'cluster_id'),
        )

    def __str__(self):
        return self.accession

    @property
    def accession(self):
        return f'{self.sample}:{self.cluster_id}'

    def get_sequence(self, fasta_format=False):
        with self.sample.get_assembly_path().open('rb') as asm:
            asm.seek(self.seq_offset)
            seq = b''
            for line in asm:
                if line.startswith(b'>'):
                    break
                seq += line.strip()
                if len(seq) > self.seq_bytes:
                    raise RuntimeError(
                        f'Unexpectedly, sequence of {self} has a length more '
                        f'than {self.seq_bytes}'
                    )

        if fasta_format:
            head = f'>{self}\n'
        else:
            head = ''
        return head + seq.decode()

    @classmethod
    def load(cls, verbose=False):
        """ Load contig cluster data for all samples """
        for i in Sample.objects.all():
            if i.assembly_ok and i.mapping_ok:
                log.info(f'sample {i} has assembly/mapping, skipping')
                continue
            error = cls.load_sample(i, verbose=verbose)
            if error is None:
                log.info(f'contig cluster loaded: {i}')
            else:
                return (i, *error)

    @classmethod
    @atomic
    def load_sample(cls, sample, limit=None, verbose=False):
        """
        import assembly/coverage data for one sample

        limit - limit to that many contig clusters, for testing only
        """
        # assumes that contigs are ordered the same in objs and cov
        objs = cls.from_sample(sample, limit=limit, verbose=verbose)
        cov = cls.read_coverage(sample)
        cls.objects.bulk_create(cls._join_cov_data(objs, cov))
        sample.assembly_ok = True
        sample.mapping_ok = True
        sample.save()

    @classmethod
    def from_sample(cls, sample, limit=None, verbose=False):
        """
        Generate contig cluster instances for given sample
        """
        print_count = verbose and sys.stdout.isatty()

        with sample.get_assembly_path().open('r') as asm:
            os.posix_fadvise(asm.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            if verbose:
                print(f'reading {asm.name} ...')
            obj = None
            count = 0
            pos = 0
            end = EOF()
            for line in chain(asm, [end]):
                if limit and count >= limit:
                    break
                pos += len(line)
                if line is end or line.startswith('>'):
                    if obj is not None:
                        obj.seq_bytes = pos - obj.seq_offset
                        # obj.full_clean()  # way too slow
                        if print_count:
                            print(f'  {count} contig clusters found', end='\r')
                        yield obj
                        count += 1

                    if line is end:
                        break

                    obj = cls(
                        cluster_id=line.rstrip()[1:],
                        sample=sample,
                        seq_offset=pos,
                    )

        if verbose:
            print(f'{count} contig clusters loaded')

    @classmethod
    def read_coverage(cls, sample, verbose=False):
        """
        read and coverage data
        """
        print_count = verbose and sys.stdout.isatty()
        rpkm_cols = ['Name', 'Length', 'Bases', 'Coverage', 'Reads', 'RPKM',
                     'Frags', 'FPKM']

        with sample.get_coverage_path('rpkm').open('r') as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            # 1. read header
            for line in f:
                if not line.startswith('#'):
                    raise ValueError(
                        '{sample}/{f}: expected header but got: {line}'
                    )
                row = line.strip().lstrip('#').split('\t')
                if row[0] == 'Name':
                    if not row == rpkm_cols:
                        raise ValueError(
                            '{sample}/{f}: unexpected column names'
                        )
                    # table starts
                    break

                # expect key/value pairs
                key, value = row
                if key == 'Reads':
                    sample.read_count = value
                elif key == 'Mapped':
                    sample.reads_mapped = value
                elif key == 'RefSequences':
                    sample.num_ref_sequences = value

            # 2. read rows
            count = 0
            for line in f:
                yield line.rstrip().split('\t')
                count += 1
                if print_count and count % 1000 == 0:
                    print(f'  {count} coverage rows processed', end='\r')

    @classmethod
    def _join_cov_data(cls, objs, cov):
        """ populate instances with coverage data """
        for obj, row in zip(objs, cov):
            if obj.cluster_id != row[0]:  # check name
                raise RuntimeError('asm and cov data is out of order')

            obj.length = row[1]
            obj.bases = row[2]
            obj.coverage = row[3]
            obj.reads_mapped = row[4]
            obj.rpkm = row[5]
            obj.frags_mapped = row[6]
            obj.fpkm = row[7]

            yield obj


class ReadLibrary(Model):
    sample = models.OneToOneField(
        'Sample',
        on_delete=models.CASCADE,
        related_name='reads',
    )
    fwd_qc0_fastq = DataPathField(base='READS')
    rev_qc0_fastq = DataPathField(base='READS')
    fwd_qc1_fastq = DataPathField(base='READS')
    rev_qc1_fastq = DataPathField(base='READS')
    raw_read_count = models.PositiveIntegerField(**opt)
    qc_read_count = models.PositiveIntegerField(**opt)

    def __str__(self):
        return self.sample.accession

    @classmethod
    @atomic
    def sync(cls, no_counts=True, raise_on_error=False):
        if not no_counts:
            raise NotImplementedError('read counting is not yet implemented')

        for i in Sample.objects.filter(reads=None):
            obj = cls.from_sample(i)
            try:
                obj.full_clean()
            except ValidationError as e:
                if raise_on_error:
                    raise
                else:
                    log.error(f'failed validation: {repr(obj)}: {e}')
                    continue
            obj.save()
            log.info(f'new read lib: {obj}')

    @classmethod
    def from_sample(cls, sample):
        obj = cls()
        obj.sample = sample
        fq = sample.get_fq_paths()
        obj.fwd_qc0_fastq = fq['dd_fwd']
        obj.rev_qc0_fastq = fq['dd_rev']
        obj.fwd_qc1_fastq = fq['ddtrnhnp_fwd']
        obj.rev_qc1_fastq = fq['ddtrnhnp_rev']
        return obj


class Sample(Model):
    accession = AccessionField()

    # data accounting
    assembly_ok = models.BooleanField(
        default=False,
        help_text='Contig cluster data loaded',
    )
    mapping_ok = models.BooleanField(
        default=False,
        help_text='coverage data loaded',
    )
    binning_ok = models.BooleanField(
        default=False,
        help_text='Binning data loaded',
    )
    checkm_ok = models.BooleanField(
        default=False,
        help_text='Binning stats loaded',
    )

    # mapping data (from ASSEMBLIES/COVERAGE)
    read_count = models.PositiveIntegerField(
        **opt,
        help_text='number of reads used for mapping',
    )
    reads_mapped = models.PositiveIntegerField(
        **opt,
        help_text='number of reads mapped to assembly',
    )
    num_ref_sequences = models.PositiveIntegerField(
        **opt,
        help_text='RefSequences number in coverage file header',
    )

    def __str__(self):
        return self.accession

    @classmethod
    @atomic
    def sync(cls, sample_list='sample_list.txt', **kwargs):
        src = settings.GLAMR_DATA_ROOT / sample_list
        with open(src) as f:
            seen = []
            for line in f:
                obj, isnew = cls.objects.get_or_create(
                    accession=line.strip()
                )
                seen.append(obj.pk)
                if isnew:
                    log.info(f'new sample: {obj}')

        not_in_src = cls.objects.exclude(pk__in=seen)
        if not_in_src.exists():
            log.warning(f'Have {not_in_src.count()} extra samples in DB not '
                        f'found in {src}')

    @classmethod
    def status(cls):
        if not cls.objects.exists():
            print('no samples in database yet')
            return

        print(' ' * 10, 'assemblyl', 'mapping', 'bins', 'checkm', sep='\t')
        for i in cls.objects.all():
            print(
                f'{i}:',
                'OK' if i.assembly_ok else '',
                'OK' if i.mapping_ok else '',
                'OK' if i.binning_ok else '',
                'OK' if i.checkm_ok else '',
                sep='\t'
            )

    @classmethod
    def status_long(cls):
        if not cls.objects.exists():
            print('no samples in database yet')
            return

        print(' ' * 10, 'cont cl', 'MAX', 'MET93', 'MET97', 'MET99', sep='\t')
        for i in cls.objects.all():
            print(
                f'{i}:',
                i.contigcluster_set.count(),
                i.binmax_set.count(),
                i.binmet93_set.count(),
                i.binmet97_set.count(),
                i.binmet99_set.count(),
                sep='\t'
            )

    def load_bins(self):
        if not self.binning_ok:
            with atomic():
                Bin.import_sample_bins(self)
                self.binning_ok = True
                self.save()
        if self.binning_ok and not self.checkm_ok:
            with atomic():
                CheckM.import_sample(self)
                self.checkm_ok = True
                self.save()

    @atomic
    def delete_bins(self):
        with atomic():
            qslist = [self.binmax_set, self.binmet93_set, self.binmet97_set,
                      self.binmet99_set]
            for qs in qslist:
                print(f'{self}: deleting {qs.model.method} bins ...', end='',
                      flush=True)
                counts = qs.all().delete()
                print('\b\b\bOK', counts)
            self.binning_ok = False
            self.checkm_ok = False  # was cascade-deleted
            self.save()

    def get_assembly_path(self):
        return (
            settings.GLAMR_DATA_ROOT / 'ASSEMBLIES' / 'MERGED'
            / (self.accession + '_MCDD.fa')
        )

    def get_coverage_path(self, filetype='rpkm'):
        fnames = {
            'rpkm': f'{self.accession}_READSvsCONTIGS.rpkm',
            'max': f'{self.accession}_MAX_coverage.txt',
            'met': f'{self.accession}_MET_coverage.txt',
        }
        base = settings.GLAMR_DATA_ROOT / 'ASSEMBLIES' / 'COVERAGE'
        try:
            return base / fnames[filetype.casefold()]
        except KeyError as e:
            raise ValueError(f'unknown filetype: {filetype}') from e

    def get_fq_paths(self):
        base = settings.GLAMR_DATA_ROOT / 'READS'
        fname = f'{self.accession}_{{infix}}.fastq.gz'
        return {
            infix: base / fname.format(infix=infix)
            for infix in ['dd_fwd', 'dd_rev', 'ddtrnhnp_fwd', 'ddtrnhnp_rev']
        }

    def get_checkm_stats_path(self):
        return (settings.GLAMR_DATA_ROOT / 'BINS' / f'{self.accession}_CHECKM'
                / 'storage' / 'bin_stats.analyze.tsv')


class Taxonomy(Model):
    SOURCE_DB_CHOICES = (
        (0, 'ncbi'),
    )
    accession = models.CharField(
        max_length=32,
        unique=True,
        verbose_name='taxonomy ID',
    )
    source_db = models.PositiveSmallIntegerField(
        choices=SOURCE_DB_CHOICES,
    )
    rank = models.CharField(max_length=32)
    name = models.CharField(max_length=255)
    name_type = models.CharField(
        max_length=32, blank=True,
        help_text='NCBI taxnonomy name type',
    )
    parent = models.ForeignKey('self', **fk_opt)

    class Meta:
        unique_together = (
            ('source_db', 'rank', 'name', 'name_type'),
        )
        verbose_name_plural = 'taxa'


class UniRef100(Model):
    # UNIREF100	NAME	LENGTH	UNIPROT_IDS	UNIREF90	TAXON_IDS
    # LINEAGE	SIGALPEP	TMS	DNA	METAL	TCDB	LOCATION
    # COG	PFAM	TIGR	GO	IPR	EC	KEGG	RHEA	BIOCYC
    # REACTANTS	PRODUCTS	TRANS_CPD
    accession = models.CharField(  # UNIREF1100
        max_length=32,
        unique=True,
        verbose_name='UniRef100 accession',
    )
    protein_name = models.CharField(max_length=32, **ch_opt)  # NAME
    taxonomic_lineage_id = models.CharField(max_length=32, **ch_opt)
    taxonomic_lineage_species = models.CharField(max_length=32, **ch_opt)
    organism = models.CharField(max_length=32, **ch_opt)
    dna_binding = models.CharField(max_length=32, **ch_opt)
    metal_binding = models.CharField(max_length=32, **ch_opt)
    signal_peptide = models.CharField(max_length=32, **ch_opt)
    transmembrane = models.CharField(max_length=32, **ch_opt)
    subcellular_location = models.CharField(max_length=32, **ch_opt)
    tcdb = models.CharField(max_length=32, **ch_opt)
    cog_kog = models.CharField(max_length=32, **ch_opt)
    pfam = models.CharField(max_length=32, **ch_opt)
    tigrfams = models.CharField(max_length=32, **ch_opt)
    kegg = models.CharField(max_length=32, **ch_opt)
    gene_ontology = models.CharField(max_length=32, **ch_opt)
    interpro = models.CharField(max_length=32, **ch_opt)
    ec_number = models.CharField(max_length=32, **ch_opt)


def load_all(**kwargs):
    """
    Load all data

    assumes an empty DB.
    """
    Sample.sync(**kwargs)
    # ReadLibrary.sync()
    ContigCluster.load(verbose=kwargs.get('verbose', False))
    Bin.import_all()
    CheckM.import_all()
