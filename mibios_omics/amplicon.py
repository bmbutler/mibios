"""
support for the amplicon pipeline
"""
from itertools import groupby, islice
from pathlib import Path
import random
import shutil
from statistics import mean, mode, stdev
import subprocess
import tempfile

from .utils import parse_fastq

ALIGNMENT_MODE_THRESHOLD = 0.8
FLIP_THRESHOLD = 0.8
FWD = 'fwd'
REV = 'rev'

REF_ALIGN = {
    '16S': '/geomicro/data2/heinro/work/glamr-16S-test/silva.nr_v138.align.10k'
}

MOTHUR_SINGLE = """
set.dir(output={cwd}, inputdir={cwd})
set.current(processors={threads})
summary.seqs(fasta={fasta})
align.seqs(reference={ref_align})
summary.seqs()
"""

MOTHUR_PAIRED = """
set.dir(output={cwd}, inputdir={cwd})
set.current(processors={threads})
make.contigs(ffastq={fwd_fastq}, rfastq={rev_fastq})
summary.seqs(contigsreport=current)
summary.seqs()
align.seqs(reference={ref_align})
summary.seqs(alignreport=fwd.trim.contigs.align.report)
summary.seqs()
"""

MOTHUR_PRIMERS = """
align.seqs(fasta={fasta},reference={ref_align},processors=1)
summary.seqs()
"""

PRIMERS = (
    # ('name', 'gene', 'region', 'fwd/rev', "sequence-5'-to-3'", start, end),
    ('357F', '16S', 'V3-V5', FWD, 'CCTACGGGAGGCAGCAG', 6388, 6426),
    ('926R', '16S', 'V3-V5', REV, 'CCGTCAATTCMTTTRAGT', 36011, 37426),
    ('SSU_F04', '18S', '', FWD, None, None, None),
    ('SSU_R22', '18S', '', REV, None, None, None),
    ('S-D-Bact-0341-b-S-17', '16S', 'V3-V4', FWD, None, None, None),
    ('S-D-Bact-0785-a-A-21', '16S', 'V3-V4', REV, None, None, None),
    ('515F-C', '16S', 'V4', FWD, 'GTGCCAGCMGCCGCGGTAA', 11895, 13861),
    ('806R', '16S', 'V4', REV, 'GGACTACHVGGGTWTCTAAT', 23446, 25318),
    # '': ('', '', '', None),
)


def quick_annotation(analysis_results, gene):
    """ annotate alignment from quick analysis """
    START = 'start'
    END = 'end'
    # get sorted list of primer/regions intervals
    primers = prep_primer_info(gene)
    regions = get_region_info()[gene]
    pos = []
    for name, (_, _, _, _, start, end) in primers.items():
        if start is None or end is None:
            continue
        pos.append((start, START, name))
        pos.append((end, END, name))
    for name, (start, end) in regions.items():
        if start is None or end is None:
            continue
        pos.append((start, START, name))
        pos.append((end, END, name))
    pos = sorted(pos)

    if analysis_results['alignment_mode_ratio'] >= ALIGNMENT_MODE_THRESHOLD:
        start, end = analysis_results['start_end_mode']
    else:
        print('[NOTICE] Using mean alignment coordinates')
        start = analysis_results['start_mean']
        end = analysis_results['end_mean']

    annotation = []
    for i, what, who in pos:
        if start <= i <= end:
            annotation.append((i, what, who))

    return annotation


def quick_analysis(fastq_dir, gene, keep=False):
    """
    Run a preliminary analysis of a subset of data

    :param Path fastq_file_base:
        Path plus base filename, the common portion of the two fastq files in
        case of paired-end reads delivered in two files
    :param str gene: Name of amplicon target gene
    :param bool keep:
        Keep copy of all intermediate data under current working directory.

    Returns dict with results.
    """
    results = {}

    # heuristically and most simply check single or paired-end
    firstfiles = sorted(fastq_dir.glob('*.fastq'))[:2]
    if len(firstfiles) == 0:
        raise RuntimeError(f'no *.fastq files found in: {fastq_dir}')
    elif len(firstfiles) == 1:
        paired = False
    else:
        # got two files
        file1, file2 = firstfiles
        if file1.stat().st_size == file2.stat().st_size:
            paired = True  # what are the chances this goes wrong?
        else:
            paired = False
    results['paired'] = paired

    # read length, overlap, target
    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        # make file names of the sub-sampled data
        if paired:
            fwd = tmpd / 'fwd.fastq'
            rev = tmpd / 'rev.fastq'
        else:
            fwd = tmpd / 'fwd.fasta'
            rev = None

        totals0, readlen = make_sub_sample_fastq(fastq_dir, fwd, rev)
        results.update(**readlen)
        mob = tmpd / 'mob'
        if paired:
            mob.write_text(MOTHUR_PAIRED.format(
                cwd=tmpd,
                threads='16',
                fwd_fastq=fwd,
                rev_fastq=rev,
                ref_align=REF_ALIGN[gene],
            ))
        else:
            mob.write_text(MOTHUR_SINGLE.format(
                cwd=tmpd,
                threads='16',
                fasta=fwd,
                ref_align=REF_ALIGN[gene],
            ))
        cmd = ['mothur', str(mob)]
        proc = subprocess.run(cmd, cwd=tmpd)
        if proc.returncode:
            input(f'ERROR running mothur -- press <enter> to clean up '
                  f'or first check tempdir: {tmpd}')
            raise RuntimeError(
                f'failed running mothur -- exit status: {proc.returncode}'
            )

        # get stats about the mean contig from contigsreport
        if paired:
            with list(tmpd.glob('mothur.*.logfile'))[0].open() as logfile:
                for line in logfile:
                    if 'summary.seqs(contigsreport=' in line:
                        break
                else:
                    raise RuntimeError('no contigs report found in mothur log')
                for line in logfile:
                    if line.startswith('Mean:'):
                        _, *mean_contig = line.split()
                        break
                else:
                    raise RuntimeError('no mean contig stats in mothur log?')
            contig_fields = ('Length Overlap_Length Overlap_Start Overlap_End '
                             'MisMatches Num_Ns')
            results['mean_contig'] = \
                dict(zip(contig_fields.split(), mean_contig))
        else:
            results['mean_contig'] = None

        # get alignment stats
        if paired:
            summary = tmpd / 'fwd.trim.contigs.summary'
        else:
            summary = tmpd / 'fwd.summary'
        with summary.open() as ifile:
            if not ifile.readline().startswith('seqname'):
                raise RuntimeError(f'bad header in summary file: {ifile}')
            starts = []
            ends = []
            for line in ifile:
                _, start, end, *_ = line.rstrip('\n').split('\t')
                starts.append(int(start))
                ends.append(int(end))
        totals = len(starts)
        if totals0 != totals:
            print(
                f'WARNING: mothur swallowed some reads? Expected {totals0} '
                f'sub-sampled reads but {totals} got aligned'
            )

        almode = mode(zip(starts, ends))
        results['start_end_mode'] = almode
        mode_count = sum((1 for i, j in zip(starts, ends) if (i, j) == almode))
        results['alignment_mode_ratio'] = mode_count / totals
        results['start_mean'] = mean(starts)
        results['start_stdev'] = stdev(starts)
        results['end_mean'] = mean(ends)
        results['end_stdev'] = stdev(ends)

        # is the alignment flipped?
        if paired:
            accnos = tmpd / 'fwd.trim.contigs.flip.accnos'
        else:
            accnos = tmpd / 'fwd.flip.accnos'
        with accnos.open() as ifile:
            flipped = 0
            for _ in ifile:
                flipped += 1
        results['flip_ratio'] = flipped / totals

        if keep:
            dest = Path() / f'tmp-data-{fastq_dir.name}'
            dest_num = None
            while dest.exists():
                if dest_num is None:
                    dest_num = 0
                    dest = dest.with_name(dest.name + f'.{dest_num}')
                else:
                    dest_num += 1
                    dest = dest.with_suffix(f'.{dest_num}')
            shutil.move(tmpd, dest)
            print(f'Mothur output copied to: {dest}')

    return results


def sample_reads(fastq_dir, paired_end, sample_size=1000, seed=None):
    """
    Prepare a sub-sampled fastq data set

    :param Path fastq_dir:
        Path to firectory with .fastq files
    :param file-like outfile1:
        Output file for first read direction
    :param file-like outfiler2:
        Output file for second read direction, if this is None, then single-end
        reads are assumed
    :param int/float sample_size:
        If this is an integer larger than 1, then this is the number of
        reads/read pairs to sample, if it is a float 0<sample_size<=1, then it
        is the approximate sampling rate

    Yields single or pair sampled fastq records
    """
    if sample_size <= 0:
        raise ValueError('sample_size must be int or float > 0')

    if sample_size > 1 and isinstance(sample_size, float):
        raise ValueError(
            'if sample size is larger than 1, then it must be given as int'
        )

    # paired-end file grouping assumes that file names sort just so that pairs
    # appear consequtivly
    files = sorted(fastq_dir.glob('*.fastq'))
    zip_args = [iter(files)] * (2 if paired_end else 1)
    files = list(zip(*zip_args))

    # estimate bytes/read
    data = ''
    for i, *_ in files:
        with i.open() as ifile:
            data = ifile.read(1024 * 1024)
        if len(data) >= 1024 * 1024:
            break
    else:
        raise RuntimeError('did not get any test data')
    if not data:
        raise RuntimeError('did not get enough test data')
    bytes_per_read = 1024 * 1024 / len(data.splitlines()) / 4  # 4 per fastq
    del data

    # get total size for one read direction
    total_size = 0  # in bytes
    for i, *_ in files:
        total_size += i.stat().st_size

    total_read_count = int(total_size / bytes_per_read)

    if sample_size <= 1:
        # get sample size from fraction
        sample_size = int(sample_size / total_read_count)

    if sample_size > total_read_count:
        raise ValueError('sample size is larger than estimated read count')

    # generate indices of samples reads
    random.seed(seed)
    sample = set()
    for _ in range(sample_size):
        while True:
            n = random.randrange(total_read_count)
            if n not in sample:
                # as sample_size gets closer to read count, getting out of this
                # while loop will take longer
                sample.add(n)
                break
    sample = iter(sorted(sample))

    zero_pos_idx = 0
    seqids = set()
    dupe_count = 0
    for file_set in files:
        file_set = [i.open() for i in file_set]
        for idx in sample:
            pos = int((idx - zero_pos_idx) * bytes_per_read)
            for i in file_set:
                i.seek(pos)

            try:
                fastqs = tuple((
                    # get the first fastq record we find after pos
                    list(
                        islice(parse_fastq(i, skip_initial_trash=True), 1, 2)
                    )[0]
                    for i in file_set
                ))
            except IndexError:
                # seeked into or beyond last record, goto next set of files
                for i in file_set:
                    i.close()
                break
            else:
                seqid = fastqs[0]['head'].split()[0].removeprefix('@')
                if seqid in seqids:
                    # FIXME: unsure why we get these duplicates
                    # print(f'WARNING: a dupe: {seqid=} {idx=} {pos=}')
                    dupe_count += 1
                    continue
                else:
                    seqids.add(seqid)
                yield fastqs
        # This sets the base in read index terms from which we calculate file
        # position for the next file(set), a bit rough (leads to some
        # undersampling relative to sample_size) but good enough
        zero_pos_idx = idx

    if dupe_count:
        print('DEBUG: caught sampling of duplicate reads: {dupe_count}')


def make_sub_sample_fastq(fastq_dir, outfile1, outfile2=None,
                          sample_size=10000):
    """
    Writes a sub sample of fastq data to the provided files

    Returns a tuple consisting of the actual number of reads samples (which may
    be lower than the given sample_size) and the mode of the read lengths of
    the sampled reads (for paired data forward reads only, reverse reads are
    assumed to be of equal lengths.)
    """
    if outfile2 is None:
        paired_end = False
    else:
        paired_end = True

    ofiles = [outfile1]
    if outfile2:
        ofiles.append(outfile2)

    ofiles = [Path(i) if isinstance(i, str) else i for i in ofiles]
    ofiles = [i.open('w') for i in ofiles]

    read_lengths = []
    for records in sample_reads(fastq_dir, paired_end, sample_size):
        for n, (ofile, rec) in enumerate(zip(ofiles, records)):
            if paired_end:
                ofile.write(f"@{rec['head']}\n{rec['seq']}\n+\n{rec['qual']}\n")  # noqa: E501
            else:
                # write as fasta
                ofile.write(f">{rec['head']}\n{rec['seq']}\n")

            if n == 0:
                read_lengths.append(len(rec['seq']))

    for i in ofiles:
        i.close()

    return len(read_lengths), {
        'read_length_mean': mean(read_lengths),
        'read_length_stdev': stdev(read_lengths),
    }


def prep_primer_info(gene=None):
    """
    verify PRIMERS data structure and convert to dict

    If the gene parameter is given, then only return primers for the given
    target gene.
    """
    select_gene = gene
    primers = {}
    for name, gene, region, direct, seq, start, end in PRIMERS:
        if name in primers:
            raise ValueError(f'duplicate primer name: {name}')
        if direct not in (FWD, REV):
            raise ValueError(f'direction must be either "{FWD}" or "{REV}"')
        if start is None:
            if end is not None:
                raise ValueError('have no start but end?')
        else:
            if end is None:
                raise ValueError('have start but no end?')
            if start >= end:
                raise ValueError('start should be less then end')

        if select_gene is not None:
            if gene != select_gene:
                continue
        primers[name] = (gene, region, direct, seq, start, end)
    return primers


def locate_primers():
    """
    locate pcr primer sequences in reference alignments
    """
    results = []
    primers = prep_primer_info()
    for gene in REF_ALIGN.keys():
        primer_seqs = {}
        for name, (gene0, _, _, seq, start, end) in primers.items():
            if gene0 != gene:
                continue
            if seq is None:
                print(f'primer sequence missing: {gene0} / {name}')
                continue
            primer_seqs[name] = seq, start, end

        with tempfile.TemporaryDirectory() as tmpd:
            tmpd = Path(tmpd)
            fasta = tmpd / 'primers.fasta'
            fasta.write_text(''.join([
                f'>{name}\n{seq}\n'
                for name, (seq, _, _) in primer_seqs.items()
            ]))
            mob = tmpd / 'mob'
            mob.write_text(MOTHUR_PRIMERS.format(
                fasta=fasta,
                ref_align=REF_ALIGN[gene],
            ))
            proc = subprocess.run(
                ['mothur', str(mob)],
                cwd=tmpd,
                capture_output=True,
            )
            if proc.returncode:
                print(f'ERROR: mothur exit status: {proc.returncode}')
                input(f'inspect temp dir: {tmpd} then press <enter> for raise')
                raise RuntimeError(f'failed running mothur: {vars(proc)}')

            with (tmpd / 'primers.summary').open() as infile:
                infile.readline()
                for line in infile:
                    row = line.rstrip('\n').split('\t')
                    name, start, end, nbases, _, _, numSeqs = row
                    start = int(start)
                    end = int(end)
                    nbases = int(nbases)
                    if numSeqs != '1':
                        print(f'WARNING: unexpected numSeqs value: {line}')
                    seq, start0, end0 = primer_seqs[name]
                    if nbases != len(seq):
                        print(f'WARNING: nbases vs length: '
                              f'{nbases} != {len(seq)}: {line}')
                    if start != start0:
                        print(f'WARNING: start position for {name} changed: '
                              f'{start0} -> {start}')
                    if end != end0:
                        print(f'WARNING: end position for {name} changed: '
                              f'{end0} -> {end}')

                    results.append((gene, name, start, end))

    return results


def get_region_info():
    """
    Return target region information from primer info
    """
    primers = prep_primer_info()
    regions = {}
    # sort primer data by gene/region/direction
    # then group by each region
    data = sorted(primers.values(), key=lambda x: x[:3])
    for (gene, region_name), grp in groupby(data, lambda x: x[:2]):
        grp = list(grp)
        try:
            fwd_primer, rev_primer = grp
        except ValueError:
            # group too small or too big, expect 2
            # TODO: expect primers to not be neatly paired?
            print(f'WARNING: ambiguous region info: {gene}/{region_name}')
        if fwd_primer[2] != FWD:
            raise ValueError(f'expected info for a fwd primer: {grp}')
        if rev_primer[2] != REV:
            raise ValueError('expected info for a rev primer')

        fwd_primer_end = fwd_primer[5]
        rev_primer_start = rev_primer[4]
        if fwd_primer_end is None or rev_primer_start is None:
            print(f'Coordinated missing: {gene} {region_name}')
            continue

        if gene not in regions:
            regions[gene] = {}
        regions[gene][region_name] = fwd_primer_end + 1, rev_primer_start - 1
    return regions
