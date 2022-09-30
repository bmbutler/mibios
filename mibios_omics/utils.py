from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def get_dataset_model():
    try:
        return django_apps.get_model(
            settings.OMICS_DATASET_MODEL,
            require_ready=False,
        )
    except ValueError:
        raise ImproperlyConfigured(
            "OMICS_DATASET_MODEL must be of the form "
            "'app_label.model_name'"
        )
    except LookupError:
        raise ImproperlyConfigured(
            f'OMICS_DATASET_MODEL refers to model '
            f'{settings.OMICS_DATASET_MODEL} that has not been installed'
        )


def get_sample_model():
    try:
        return django_apps.get_model(
            settings.OMICS_SAMPLE_MODEL,
            require_ready=False,
        )
    except ValueError:
        raise ImproperlyConfigured(
            "OMICS_SAMPLE_MODEL must be of the form "
            "'app_label.model_name'"
        )
    except LookupError:
        raise ImproperlyConfigured(
            f'OMICS_SAMPLE_MODEL refers to model '
            f'{settings.OMICS_SAMPLE_MODEL} that has not been installed'
        )


def get_fasta_sequence(file, offset, length, skip_header=True):
    """
    Retrieve sequence record from fasta formatted file with known offset

    parameters:
        file: file like object, opened for reading bytes
        offset: first byte of header
        length: length of data in bytes

    Returns the fasta record or sequence as bytes string.  The sequence part
    will be returned in a single line even if it was broken up into multiple
    line originally.
    """
    file.seek(offset)
    if skip_header:
        header = file.readline()[0] == b'>'
        if header[0] != b'>':
            raise RuntimeError('expected fasta header start ">" missing')
        length -= len(header)
        if length < 0:
            raise ValueError('header is longer than length')
    else:
        # not bothering with any checks here
        pass

    data = file.read(length).splitlines()
    if not skip_header:
        data.insert(1, b'\n')
    data = b''.join(data.splitlines())
    return data


class parse_fastq:
    """
    Generate fastq records from something file-like.

    usage:
        for record in parse_fastq(open('my.fastq')):
            head = record['head']
            seq = record['seq']
            qual = record['qual']

    Records must be 4 lines long.  Returns a dict, trims off newlines and
    initial @ from header.
    """
    def __init__(self, file, skip_initial_trash=False):
        """
        Setup the fastq parser

        :param bool skip_initial_trash:
            If True, then any initial lines that don't parse as fastq will be
            skipped without raising an error.  This also assumes trat file is
            seekable.

        Parsing will not be reliable if seek() is called on the underying file
        t = descriptor after next() has been called already.
        """
        self.file = file
        if skip_initial_trash:
            pos = self.file.tell()
            while True:
                try:
                    self._make_record()
                except StopIteration:
                    # EOF
                    break
                except ValueError:
                    # advance one line and try again
                    self.file.seek(pos)
                    if self.file.readline():
                        pos = self.file.tell()
                    else:
                        # EOF
                        break
                else:
                    # there is a good record at pos
                    self.file.seek(pos)
                    break

    def __iter__(self):
        return self

    def __next__(self):
        return self._make_record()

    def _get_line(self):
        return next(self.file).rstrip('\n')

    def _make_record(self):
        """
        Build the record from buffer and next four lines

        Raises ValueError if checks for correct fastq format fail.
        """
        head = self._get_line()  # StopIteration here will bubble up
        if not head.startswith('@'):
            raise ValueError(f'expected @ at start of line: {head}')

        try:
            head = head.removeprefix('@')
            seq = self._get_line()
            if not seq:
                raise ValueError(f'sequence missing: {seq}')
            plus = self._get_line()
            if not plus.startswith('+'):
                raise ValueError(f'expected "+" at start of line: {plus}')
            qual = self._get_line()
            if len(qual) != len(seq):
                raise ValueError('sequence and quality differ in length')
            return dict(head=head, seq=seq, qual=qual)
        except StopIteration:
            # got a header but further lines missing
            raise ValueError('file ends with incomplete record')
