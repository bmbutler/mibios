from logging import getLogger
from pathlib import Path
import re
from subprocess import PIPE, Popen, TimeoutExpired

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.transaction import atomic, set_rollback

from mibios.data import TableConfig
from mibios.umrad.fields import AccessionField
from mibios.umrad.model_utils import (
    digits, opt, ch_opt, fk_req, fk_opt, uniq_opt, Model,
)
from mibios.umrad.models import (CompoundRecord, FuncRefDBEntry, TaxID, Taxon,
                                 UniRef100)
from mibios.umrad.utils import ProgressPrinter

from . import managers, get_sample_model, sra
from .amplicon import get_target_genes, quick_analysis, quick_annotation
from .fields import DataPathField
from .utils import get_fasta_sequence


log = getLogger(__name__)


class AbstractAbundance(Model):
    """
    abundance vs <something>

    With data from the Sample_xxxx_<something>_VERSION.txt files
    """
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    scos = models.DecimalField(**digits(12, 2))
    rpkm = models.DecimalField(**digits(12, 2))
    # lca ?

    class Meta(Model.Meta):
        abstract = True


class AbstractSample(Model):
    TYPE_AMPLICON = 'amplicon'
    TYPE_METAGENOME = 'metagenome'
    TYPE_METATRANS = 'metatranscriptome'
    SAMPLE_TYPES_CHOICES = (
        (TYPE_AMPLICON, TYPE_AMPLICON),
        (TYPE_METAGENOME, TYPE_METAGENOME),
        (TYPE_METATRANS, TYPE_METATRANS),
    )

    sample_id = models.CharField(
        max_length=256,
        # TODO: make required
        **uniq_opt,
        help_text='internal sample accession',
    )
    tracking_id = models.CharField(
        # length of md5 checksum, 128 bit as hex string
        max_length=32,
        **uniq_opt,
        help_text='internal uniform hex id',
    )
    sample_name = models.CharField(
        max_length=64,
        help_text='sample ID or name as given by study',
    )
    dataset = models.ForeignKey(
        settings.OMICS_DATASET_MODEL,
        **fk_opt,
    )
    sample_type = models.CharField(
        max_length=32,
        choices=SAMPLE_TYPES_CHOICES,
        **opt,
    )
    has_paired_data = models.BooleanField(**opt)
    sra_accession = models.CharField(max_length=16, **ch_opt, help_text='SRA accession')  # noqa: E501
    amplicon_target = models.CharField(max_length=16, **ch_opt)
    fwd_primer = models.CharField(max_length=32, **ch_opt)
    rev_primer = models.CharField(max_length=32, **ch_opt)

    # sample data accounting flags
    meta_data_loaded = models.BooleanField(
        # this flag is not managed in mibios.omics but by downstream
        # implementers of the abstract sample model
        default=False,
        help_text='meta data successfully loaded',
    )
    metag_pipeline_reg = models.BooleanField(
        default=False,
        help_text='is registered in metagenomic pipeline, has tracking ID',
    )
    contig_fasta_loaded = models.BooleanField(
        default=False,
        help_text='contig fasta data loaded',
    )
    gene_fasta_loaded = models.BooleanField(
        default=False,
        help_text='gene fasta data loaded',
    )
    contig_abundance_loaded = models.BooleanField(
        default=False,
        help_text='contig abundance/rpkm data data loaded',
    )
    gene_abundance_loaded = models.BooleanField(
        default=False,
        help_text='gene abundance/rpkm data loaded',
    )
    gene_alignment_hits_loaded = models.BooleanField(
        default=False,
        help_text='gene alignment hits to UniRef100 loaded',
    )
    binning_ok = models.BooleanField(
        default=False,
        help_text='Binning data loaded',
    )
    checkm_ok = models.BooleanField(
        default=False,
        help_text='Binning stats loaded',
    )
    genes_ok = models.BooleanField(
        default=False,
        help_text='Gene data and coverage loaded',
    )
    proteins_ok = models.BooleanField(
        default=False,
        help_text='Protein data loaded',
    )
    tax_abund_ok = models.BooleanField(
        default=False,
        help_text='Taxon abundance data loaded',
    )
    func_abund_ok = models.BooleanField(
        default=False,
        help_text='Function abundance data loaded',
    )
    comp_abund_ok = models.BooleanField(
        default=False,
        help_text='Compound abundance data loaded',
    )

    analysis_dir = models.CharField(
        max_length=256,
        **opt,
        help_text='path to results of analysis, relative to OMICS_DATA_ROOT',
    )
    # mapping data / header items from bbmap output:
    read_count = models.PositiveIntegerField(
        **opt,
        help_text='number of read pairs (post-QC) used for assembly mapping',
    )
    reads_mapped_contigs = models.PositiveIntegerField(
        **opt,
        help_text='number of reads mapped to contigs',
    )
    reads_mapped_genes = models.PositiveIntegerField(
        **opt,
        help_text='number of reads mapped to genes',
    )

    objects = managers.SampleManager()

    class Meta:
        abstract = True
        unique_together = (
            ('dataset', 'sample_id'),
        )

    def __str__(self):
        return self.sample_id or self.tracking_id or super().__str__()

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

    def get_metagenome_path(self):
        if self.tracking_id is None:
            raise RuntimeError(f'sample {self} has no tracking id')
        return settings.OMICS_DATA_ROOT / 'data' / 'omics' / 'metagenomes' \
            / self.tracking_id

    def get_fq_paths(self):
        base = settings.OMICS_DATA_ROOT / 'READS'
        fname = f'{self.accession}_{{infix}}.fastq.gz'
        return {
            infix: base / fname.format(infix=infix)
            for infix in ['dd_fwd', 'dd_rev', 'ddtrnhnp_fwd', 'ddtrnhnp_rev']
        }

    def get_checkm_stats_path(self):
        return (settings.OMICS_DATA_ROOT / 'BINS' / 'CHECKM'
                / f'{self.accession}_CHECKM' / 'storage'
                / 'bin_stats.analyze.tsv')

    @atomic
    def load_omics_data(self, dry_run=False):
        """
        load all omics data for this sample from data source

        Assumes no existing data
        """
        Contig.loader.load_sample(self)
        Gene.loader.load_sample(self)
        FuncAbundance.loader.load_sample(self)
        CompoundAbundance.loader.load_sample(self)
        TaxonAbundance.objects.load_sample(self)
        set_rollback(dry_run)

    def get_fastq_prefix(self):
        """
        Prefix for fastq filenames for this sample
        """
        if self.sample_id is None:
            part1 = f'pk{self.pk}'
        else:
            part1 = self.sample_id

        # upstream-given name w/o funny stuff
        part2 = re.sub(r'\W+', '', self.sample_name)
        return f'{part1}-{part2}'

    def get_fastq_base(self):
        """ Get directory plus base name of fastq files """
        base = self.dataset.get_fastq_path(self.sample_type)
        return base / self.get_fastq_prefix()

    def download_fastq(self, exist_ok=False, verbose=False):
        """
        download fastq files from SRA

        Returns a triple od SRA run accession, platform, list of files.  The
        number of files in the list indicates if the data is single-end or
        paired-end.  There will be either two files or one file respectively.
        """
        fastq_base = self.get_fastq_base()
        run, platform, is_paired_end = self.get_sra_run_info()

        # if output files exist, then fastq-dump will give a cryptic error
        # message, so dealing with already existing files here:
        files = list(fastq_base.parent.glob(f'{fastq_base.name}*'))
        if files:
            if exist_ok:
                if verbose:
                    print('Some destination file(s) exist already:')
                    for i in files:
                        print(f'exists: {i}')
                if is_paired_end and len(files) >= 2:
                    # TODO: check existing file via MD5?
                    pass
                elif not is_paired_end and len(files) >= 1:
                    # TODO: check existing file via MD5?
                    pass
                else:
                    # but some file missing, trigger full download
                    files = []
            else:
                files = "\n".join([str(i) for i in files])
                raise RuntimeError(f'file(s) exist:\n{files}')

        if not files:
            files = sra.download_fastq(
                run['accession'],
                dest=fastq_base,
                exist_ok=exist_ok,
                verbose=verbose,
            )

        # Check that file names follow SRA fasterq-dump output file naming
        # conventions as expected
        if len(files) == 1:
            if self.has_paired_data:
                print(f'WARNING: {self}: {self.has_paired_data=} / expected '
                      f'two files, got: {files}')
            if files[0].name != self.get_fastq_prefix() + '.fastq':
                raise RuntimeError(f'unexpected single-end filename: {files}')
        elif len(files) == 2:
            if not self.has_paired_data and self.has_paired_data is not None:
                print(f'WARNING: {self}: {self.has_paired_data=} / expected '
                      f'single file, got: {files}')
            fnames = sorted([i.name for i in files])
            if fnames[0] != self.get_fastq_prefix() + '_1.fastq':
                raise RuntimeError(f'unexpected paired filename: {fnames[0]}')
            if fnames[1] != self.get_fastq_prefix() + '_2.fastq':
                raise RuntimeError(f'unexpected paired filename: {fnames[0]}')
        elif len(files) == 3:
            # TODO: we should expect paired data with some single reads
            raise NotImplementedError
        else:
            raise RuntimeError(
                f'unexpected number of files downloaded: {files}'
            )

        # return values are suitable for Dataset.download_fastq()
        return run['accession'], platform, is_paired_end, files

    def get_sra_run_info(self):
        if self.sra_accession.startswith(('SRR', 'SRX')):
            other = self.sra_accession
        else:
            # might be ambiguous, hope for the best
            other = None
        return sra.get_run(self.biosample, other=other)

    def amplicon_test(self, dry_run=False):
        """
        Run quick amplicon/primer location test
        """
        if self.sample_type != self.TYPE_AMPLICON:
            raise RuntimeError(
                f'method is only for {self.TYPE_AMPLICON} samples'
            )

        for i in get_target_genes():
            if i in self.amplicon_target:
                gene = i
                break
        else:
            raise RuntimeError('gene target not supported')

        data = quick_analysis(
            self.get_fastq_base().parent,
            glob=self.get_fastq_prefix() + '*.fastq',
            gene=gene,
        )
        annot = quick_annotation(data, gene)
        return annot

    def assign_analysis_unit(self, create=True):
        """
        Assign sample to analysis unit, create on if needed

        Returns the AmpliconAnalysisUnit object
        """
        '''
        obj = AmpliconAnalysisUnit.objects.get_or_create(
            dataset=self.dataset,
            # TODO
        )
        # TODO: need better localization data to do this
        '''


class AbstractDataset(Model):
    """
    Abstract base model for a study or similar collections of samples

    To be used by apps that implement meta data models.
    """
    dataset_id = models.PositiveIntegerField(
        unique=True,
        help_text='internal accession to data set/study/project',
    )
    short_name = models.CharField(
        max_length=64,
        **uniq_opt,
        help_text='a short name or description, for internal use, not '
                  '(necessarily) for public display',
    )

    class Meta:
        abstract = True

    def __init__(self, *args, orphan_group=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.orphan_group = orphan_group

    _orphan_group_obj = None
    orphan_group_description = 'ungrouped samples'

    def samples(self):
        """
        Return samples in a way that also works for the orphan set
        """
        if self.orphan_group:
            return get_sample_model().objects.filter(dataset=None)
        else:
            return self.sample_set.all()

    def get_samples_url(self):
        """
        Returns mibios table interface URL for table of the study's samples
        """
        conf = TableConfig(get_sample_model())
        if self.orphan_group:
            conf.filter['dataset'] = None
        else:
            conf.filter['dataset__pk'] = self.pk
        return conf.url()

    @classmethod
    @property
    def orphans(cls):
        """
        Return the fake group of samples without a study

        The returned instance is for display purpose only, should not be saved.
        Implementing classes may want to set any useful attributes such as the
        group's name or description.
        """
        if cls._orphan_group_obj is None:
            cls._orphan_group_obj = cls(orphan_group=True)
        return cls._orphan_group_obj

    def download_fastq(self, exist_ok=False):
        """ get fastq data from SRA """
        # TODO: implement correct behaviour if exist_ok and file actually exist
        manifest = []
        for i in self.sample_set.all():
            run, platform, is_paired_end, files = i.download_fastq(exist_ok=exist_ok)  # noqa: E501
            files = [i.name for i in files]
            if len(files) == 1:
                # single reads
                read1 = files[0]
                read2 = ''
            elif len(files) == 2:
                # paired-end reads, assume file names differ by infix:
                # _1 <=> _2 following SRA fasterq-dump conventions,
                # this sorts correctly
                read1, read2 = sorted(files)
            else:
                raise ValueError('can only handle one or two files per sample')

            manifest.append((
                i.sample_id,
                i.sample_name,
                run,
                platform,
                'paired' if read2 else 'single',
                i.amplicon_target,
                read1,
                read2,
            ))

        mfile = self.get_fastq_path() / 'fastq_manifest.csv'
        with open(mfile, 'w') as out:
            for row in manifest:
                out.write('\t'.join(row))
                out.write('\n')
        print(f'manifest written to: {mfile}')

    def get_sample_type(self):
        """
        Determine sample type
        """
        stypes = set(self.sample_set.values_list('sample_type', flat=True))
        num = len(stypes)
        if num == 0:
            raise RuntimeError('data set has no samples')
        elif num > 1:
            # TODO: do we need to support this?
            raise RuntimeError('multiple types: {stypes}')
        return stypes.pop()

    def get_fastq_path(self, sample_type=None):
        """
        Get path to fastq data storage
        """
        if sample_type is None:
            sample_type = self.get_sample_type()

        if sample_type == AbstractSample.TYPE_AMPLICON:
            base = Path(settings.AMPLICON_PIPELINE_BASE)
        else:
            raise NotImplementedError

        # FIXME: use study_id, but that's currently GLAMR-specific ?
        return base / str(self.dataset_id)

    def prepare_amplicon_analysis(self):
        """
        Ensure that amplicon analysis can be run on dataset

        1. ensure fastq data is downloaded for all samples
        2. ensure every sample is in one analysis unit
        3. ensure we have info from preliminary analysis
        """
        ...


class Alignment(Model):
    """ Model for a gene vs. UniRef100 alignment hit """
    history = None
    gene = models.ForeignKey('Gene', **fk_req)
    ref = models.ForeignKey(UniRef100, **fk_req)
    score = models.PositiveIntegerField()

    loader = managers.AlignmentLoader()

    class Meta:
        unique_together = (('gene', 'ref'),)


'''
class AmpliconAnalysisUnit(Model):
    """ a collection of amplicon samples to be analysed together """
    dataset = models.ForeignKey(settings.OMICS_DATASET_MODEL, **fk_req)
    seq_platform = models.CharField(max_length=32)
    is_paired_end = models.BooleanField()
    target_gene = models.CharField(max_length=16)
    fwd_primer = models.CharField(max_length=16)
    rev_primer = models.CharField(max_length=16)
    trim_params = models.CharField(max_length=128)

    class Meta:
        unique_together = (
            ('dataset', 'seq_platform', 'is_paired_end', 'fwd_primer',
             'rev_primer',),
        )
'''


class Bin(Model):
    history = None
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
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
    def get_concrete_models(cls):
        """
        Return list of all concrete bin sub-classes/models
        """
        # The recursion stops at the first non-abstract models, so this may not
        # make sense in a multi-table inheritance setting.
        children = cls.__subclasses__()
        if children:
            ret = []
            for i in children:
                ret += i.get_concrete_models()
            return ret
        else:
            # method called on a non-parent
            if cls._meta.abstract:
                return []
            else:
                return [cls]

    @classmethod
    def get_class(cls, method):
        """
        Get the concrete model for give binning method
        """
        if cls.method is not None:
            return super().get_class(method)

        for i in cls.get_concrete_models():
            if i.method == method:
                return i

        raise ValueError(f'not a valid binning type/method: {method}')

    @classmethod
    def import_all(cls):
        """
        Import all binning data

        This class method can be called on the abstract parent Bin class and
        will then import data for all binning types.  Or it can be called on an
        concrete model/class and then will only import data for the
        corresponding binning type.
        """
        if not cls._meta.abstract:
            raise RuntimeError(
                'method can not be called by concrete bin subclass'
            )
        for i in get_sample_model().objects.all():
            cls.import_sample_bins(i)

    @classmethod
    def import_sample_bins(cls, sample):
        """
        Import all types of bins for given sample
        """
        if sample.binning_ok:
            log.info(f'{sample} has bins loaded already')
            return

        if cls._meta.abstract:
            # Bin parent class only
            with atomic():
                noerr = True
                for klass in cls.get_concrete_models():
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

        qs = Contig.objects.filter(sample=sample, contig_id__in=cids)
        kwargs = {cls._meta.get_field('members').remote_field.name: obj}
        qs.update(**kwargs)
        log.info(f'{obj} imported: {len(cids)} contigs')
        return len(cids)


class BinMAX(Bin):
    method = 'MAX'

    @classmethod
    def bin_files(cls, sample):
        """
        Generator over bin file paths
        """
        pat = f'{sample.accession}_{cls.method}_bins.*.fasta'
        path = settings.OMICS_DATA_ROOT / 'BINS' / 'MAX_BIN'
        return path.glob(pat)

    class Meta:
        verbose_name = 'MaxBin'
        verbose_name_plural = 'MaxBin bins'


class BinMetaBat(Bin):

    class Meta(Bin.Meta):
        abstract = True

    @classmethod
    def bin_files(cls, sample):
        """
        Generator over bin file paths
        """
        pat = f'{sample.accession}_{cls.method}_bins.*'
        path = settings.OMICS_DATA_ROOT / 'BINS' / 'METABAT'
        return path.glob(pat)


class BinMET93(BinMetaBat):
    method = 'MET_P97S93E300'

    class Meta:
        verbose_name = 'MetaBin 97/93'
        verbose_name_plural = 'MetaBin 97/93 bins'


class BinMET97(BinMetaBat):
    method = 'MET_P99S97E300'

    class Meta:
        verbose_name = 'MetaBin 99/97'
        verbose_name_plural = 'MetaBin 99/97 bins'


class BinMET99(BinMetaBat):
    method = 'MET_P99S99E300'

    class Meta:
        verbose_name = 'MetaBin 99/99'
        verbose_name_plural = 'MetaBin 99/99 bins'


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

    class Meta:
        verbose_name = 'CheckM'
        verbose_name_plural = 'CheckM records'

    @classmethod
    def import_all(cls):
        for i in get_sample_model().objects.all():
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
                raise ValueError(f'Bad method in stats: {binid}: {e}') from e

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


class TaxonAbundance(Model):
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    taxon = models.ForeignKey(Taxon, **fk_req, related_name='abundance')
    sum_gene_rpkm = models.DecimalField(**digits(12, 4))

    objects = managers.TaxonAbundanceManager()

    class Meta(Model.Meta):
        unique_together = (
            ('sample', 'taxon'),
        )

    def __str__(self):
        return (f'{self.taxon.name}/{self.sample.sample_name} '
                f'{self.sum_gene_rpkm}')


class CompoundAbundance(AbstractAbundance):
    """
    abundance vs. compounds

    with data from Sample_xxxxx_compounds_VERSION.txt files
    """
    ROLE_CHOICES = (
        ('r', 'REACTANT'),
        ('p', 'PRODUCT'),
        ('t', 'TRANSPORT'),
    )
    compound = models.ForeignKey(
        CompoundRecord,
        related_name='abundance',
        **fk_req,
    )
    role = models.CharField(max_length=1, choices=ROLE_CHOICES)

    loader = managers.CompoundAbundanceLoader()

    class Meta(AbstractAbundance.Meta):
        unique_together = (
            ('sample', 'compound', 'role'),
        )

    def __str__(self):
        return f'{self.compound.accession} ({self.role[0]}) {self.rpkm}'


class SequenceLike(Model):
    """
    Models for sequences as found in fasta (or similar) files
    """
    history = None
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)

    fasta_offset = models.PositiveIntegerField(
        **opt,
        help_text='offset of first byte of fasta (or similar) header, if there'
                  ' is one, otherwise first byte of sequence',
    )
    fasta_len = models.PositiveIntegerField(
        **opt,
        help_text='length of sequence record in bytes, header+sequence '
                  'including internal and final newlines or until EOF.'
    )

    objects = managers.SequenceLikeManager()

    class Meta:
        abstract = True

    def get_sequence(self, fasta_format=False, file=None,
                     original_header=False):
        """
        Retrieve sequence from file storage, optionally fasta-formatted

        To get sequences for many objects, use the to_fasta() queryset method,
        which is much more efficient than calling this method while iterating
        over a queryset.

        If fasta_format is True, then the output will be a two-line string that
        may end with a newline.  If original_header is False a new header based
        on the model instance will be generated.  If fasta_format is False, the
        return value will be a string without any newlines.
        """
        if original_header and not fasta_format:
            raise ValueError('incompatible parameters: can only ask for '
                             'original header with fasta format')
        if file is None:
            p = self._meta.managers_map['loader'].get_fasta_path(self.sample)
            fh = p.open('rb')
        else:
            fh = file

        skip_header = not fasta_format or not original_header
        try:
            seq = get_fasta_sequence(fh, self.fasta_offset, self.fasta_len,
                                     skip_header=skip_header)
        finally:
            if file is None:
                fh.close()

        seq = seq.decode()
        if fasta_format and not original_header:
            return ''.join([f'>{self}\n', seq])
        else:
            return seq


class ContigLike(SequenceLike):
    """
    Abstract parent class for sequence like data with converage info

    This is for contigs and genes but not proteins.

    Contigs and Genes have a few things in common: fasta files with sequences
    and bbmap coverage results.  This class covers those commonalities.  There
    are a few methods that need to be implemented by the children that spell
    out the differences.  Those deal with where to find the files and different
    fasta headers.
    """
    # Data from mapping / coverage:
    # decimals in bbmap output have 4 fractional places
    # FIXME: determine max_places for sure
    length = models.PositiveIntegerField(**opt)
    bases = models.PositiveIntegerField(**opt)
    coverage = models.DecimalField(decimal_places=4, max_digits=10, **opt)
    reads_mapped = models.PositiveIntegerField(**opt)
    rpkm_bbmap = models.DecimalField(decimal_places=4, max_digits=10, **opt)
    rpkm = models.FloatField(**opt)
    frags_mapped = models.PositiveIntegerField(**opt)
    fpkm_bbmap = models.DecimalField(decimal_places=4, max_digits=10, **opt)
    fpkm = models.FloatField(**opt)
    # taxon + lca from contigs file
    taxid = models.ManyToManyField(TaxID, related_name='classified_%(class)s')
    lca = models.ForeignKey(Taxon, **fk_opt)

    class Meta:
        abstract = True

    # Name of the (per-sample) id field, must be set in inheriting class
    id_field_name = None


class Contig(ContigLike):
    contig_id = AccessionField(unique=False, max_length=10)

    # bin membership
    bin_max = models.ForeignKey(BinMAX, **fk_opt, related_name='members')
    bin_m93 = models.ForeignKey(BinMET93, **fk_opt, related_name='members')
    bin_m97 = models.ForeignKey(BinMET97, **fk_opt, related_name='members')
    bin_m99 = models.ForeignKey(BinMET99, **fk_opt, related_name='members')

    loader = managers.ContigLoader()

    class Meta:
        default_manager_name = 'objects'
        unique_together = (
            ('sample', 'contig_id'),
        )

    id_field_name = 'contig_id'

    def __str__(self):
        return self.accession

    @property
    def accession(self):
        return f'{self.sample}:{self.contig_id}'

    def set_from_fa_head(self, fasta_head_line):

        # parsing ">deadbeef_123\n" -> "123"
        self.contig_id = fasta_head_line.lstrip('>').rstrip().partition('_')[2]


class FuncAbundance(AbstractAbundance):
    """
    abundance vs functions

    With data from the Sample_xxxx_functions_VERSION.txt files
    """
    function = models.ForeignKey(
        FuncRefDBEntry,
        related_name='abundance',
        **fk_req,
    )

    loader = managers.FuncAbundanceLoader()

    class Meta(Model.Meta):
        unique_together = (
            ('sample', 'function'),
        )

    def genes(self):
        """
        Queryset of associated genes
        """
        return Gene.objects.filter(
            sample=self.sample,
            besthit__function_refs=self.function,
        )


class Gene(ContigLike):
    STRAND_CHOICE = (
        ('+', '+'),
        ('-', '-'),
    )
    gene_id = AccessionField(unique=False, max_length=20)
    contig = models.ForeignKey('Contig', **fk_req)
    start = models.PositiveIntegerField()
    end = models.PositiveIntegerField()
    strand = models.CharField(choices=STRAND_CHOICE, max_length=1)
    besthit = models.ForeignKey(UniRef100, **fk_opt, related_name='gene_best')
    hits = models.ManyToManyField(UniRef100, through=Alignment,
                                  related_name='gene_hit')

    loader = managers.GeneLoader()

    class Meta:
        default_manager_name = 'objects'
        unique_together = (
            ('sample', 'gene_id'),
        )

    id_field_name = 'gene_id'

    def __str__(self):
        return self.accession

    @property
    def accession(self):
        return f'{self.sample}:{self.gene_id}'

    def set_from_fa_head(self, line, contig_id_map, **kwargs):
        # parsing prodigal info
        gene_id, start, end, strand, misc = line.lstrip('>').rstrip().split(' # ')  # noqa: E501

        if strand == '1':
            strand = '+'
        elif strand == '-1':
            strand = '-'
        else:
            raise ValueError('expected strand to be "1" or "-1"')

        # name expected to be: deadbeef_123_1
        gene_id = gene_id.partition('_')[2]  # get e.g. 123_1
        self.gene_id = gene_id
        contig_id = gene_id.partition('_')[0]  # get e.g. 123
        try:
            # contig_id_map must be a mapping from contig_ids to Contig PKs
            self.contig_id = contig_id_map[contig_id]
        except KeyError as e:
            raise RuntimeError(
                f'no such contig: {e} -- ensure contigs are loaded'
            )
        self.start = start
        self.end = end
        self.strand = strand


class NCRNA(Model):
    history = None
    sample = models.ForeignKey(settings.OMICS_SAMPLE_MODEL, **fk_req)
    contig = models.ForeignKey('Contig', **fk_req)
    match = models.ForeignKey('RNACentralRep', **fk_req)
    part = models.PositiveIntegerField(**opt)

    # SAM alignment section data
    flag = models.PositiveIntegerField(help_text='bitwise FLAG')
    pos = models.PositiveIntegerField(
        help_text='1-based leftmost mapping position',
    )
    mapq = models.PositiveIntegerField(help_text='MAPing Quality')

    class Meta:
        unique_together = (
            ('sample', 'contig', 'part'),
        )

    def __str__(self):
        if self.part is None:
            part = ''
        else:
            part = f'part_{self.part}'
        return f'{self.sample.accession}:{self.contig}{part}->{self.match}'

    @classmethod
    def get_sam_file(cls, sample):
        return (settings.OMICS_DATA_ROOT / 'NCRNA'
                / f'{sample.accession}_convsrna.sam')


class Protein(SequenceLike):
    gene = models.OneToOneField(Gene, **fk_req)

    def __str__(self):
        return str(self.gene)

    @classmethod
    def have_sample_data(cls, sample, set_to=None):
        if set_to is None:
            return sample.proteins_ok
        else:
            sample.proteins_ok = set_to
            sample.save()

    @classmethod
    def get_fasta_path(cls, sample):
        return (settings.OMICS_DATA_ROOT / 'PROTEINS'
                / f'{sample.accession}_PROTEINS.faa')

    @classmethod
    def get_load_sample_fasta_extra_kw(cls, sample):
        # returns a dict of the sample's genes pks
        qs = sample.gene_set.values_list('gene_id', 'pk')
        return dict(gene_ids=dict(qs.iterator()))

    def set_from_fa_head(self, line, **kwargs):
        if 'gene_ids' in kwargs:
            gene_ids = kwargs['gene_ids']
        else:
            raise ValueError('Expect "contig_ids" in kw args')

        # parsing prodigal info
        gene_accn, _, _ = line.lstrip('>').rstrip().partition(' # ')

        self.gene_id = gene_ids[gene_accn]


class ReadLibrary(Model):
    sample = models.OneToOneField(
        settings.OMICS_SAMPLE_MODEL,
        on_delete=models.CASCADE,
        related_name='reads',
    )
    fwd_qc0_fastq = DataPathField(base='READS')
    rev_qc0_fastq = DataPathField(base='READS')
    fwd_qc1_fastq = DataPathField(base='READS')
    rev_qc1_fastq = DataPathField(base='READS')
    raw_read_count = models.PositiveIntegerField(**opt)
    qc_read_count = models.PositiveIntegerField(**opt)

    class Meta:
        verbose_name_plural = 'read libraries'

    def __str__(self):
        return self.sample.accession

    @classmethod
    @atomic
    def sync(cls, no_counts=True, raise_on_error=False):
        if not no_counts:
            raise NotImplementedError('read counting is not yet implemented')

        for i in get_sample_model().objects.filter(reads=None):
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


class RNACentral(Model):
    history = None
    RNA_TYPES = (
        # unique 3rd column from rnacentral_ids.txt
        (1, 'antisense_RNA'),
        (2, 'autocatalytically_spliced_intron'),
        (3, 'guide_RNA'),
        (4, 'hammerhead_ribozyme'),
        (5, 'lncRNA'),
        (6, 'miRNA'),
        (7, 'misc_RNA'),
        (8, 'ncRNA'),
        (9, 'other'),
        (10, 'piRNA'),
        (11, 'precursor_RNA'),
        (12, 'pre_miRNA'),
        (13, 'ribozyme'),
        (14, 'RNase_MRP_RNA'),
        (15, 'RNase_P_RNA'),
        (16, 'rRNA'),
        (17, 'scaRNA'),
        (18, 'scRNA'),
        (19, 'siRNA'),
        (20, 'snoRNA'),
        (21, 'snRNA'),
        (22, 'sRNA'),
        (23, 'SRP_RNA'),
        (24, 'telomerase_RNA'),
        (25, 'tmRNA'),
        (26, 'tRNA'),
        (27, 'vault_RNA'),
        (28, 'Y_RNA'),
    )
    INPUT_FILE = (settings.OMICS_DATA_ROOT / 'NCRNA' / 'RNA_CENTRAL'
                  / 'rnacentral_clean.fasta.gz')

    accession = AccessionField()
    taxon = models.ForeignKey(Taxon, **fk_req)
    rna_type = models.PositiveSmallIntegerField(choices=RNA_TYPES)

    def __str__(self):
        return (f'{self.accession} {self.taxon.taxid} '
                f'{self.get_rna_type_display()}')

    @classmethod
    def load(cls, path=INPUT_FILE):
        type_map = dict(((b.casefold(), a) for a, b, in cls.RNA_TYPES))
        taxa = dict(Taxon.objects.values_list('taxid', 'pk'))

        zcat_cmd = ['/usr/bin/unpigz', '-c', str(path)]
        zcat = Popen(zcat_cmd, stdout=PIPE)

        pp = ProgressPrinter('rna central records read')
        objs = []
        try:
            print('BORK STAGE I')
            for line in zcat.stdout:
                # line is bytes
                if not line.startswith(b'>'):
                    continue
                line = line.decode()
                accn, typ, taxid2, _ = line.lstrip('>').split('|', maxsplit=4)
                # taxid2 can be multiple, so take taxid from accn (ask Teal?)
                accn, _, taxid = accn.partition('_')
                objs.append(cls(
                    accession=accn,
                    taxon_id=taxa[int(taxid)],
                    rna_type=type_map[typ.lower()],
                ))
                pp.inc()
        except Exception:
            raise
        finally:
            try:
                zcat.communicate(timeout=15)
            except TimeoutExpired:
                zcat.kill()
                zcat.communicate()
            if zcat.returncode:
                log.error(f'{zcat_cmd} returned with status {zcat.returncode}')

        pp.finish()
        log.info(f'Saving {len(objs)} RNA Central accessions...')
        cls.objects.bulk_create(objs)


class RNACentralRep(Model):
    """ Unique RNACentral representatives """
    history = None


class Sample(AbstractSample):
    """
    Placeholder model for (metagenomic?) samples
    """
    class Meta:
        swappable = 'OMICS_SAMPLE_MODEL'


class Dataset(AbstractDataset):
    """
    Placeholder model implementing a dataset
    """
    class Meta:
        swappable = 'OMICS_DATASET_MODEL'


def load_all(**kwargs):
    """
    Load all data

    assumes an empty DB.
    """
    verbose = kwargs.get('verbose', False)
    get_sample_model().sync(**kwargs)
    # ReadLibrary.sync()
    Contig.load(verbose=verbose)
    Bin.import_all()
    CheckM.import_all()
    Gene.load(verbose=verbose)
    Protein.load(verbose=verbose)
