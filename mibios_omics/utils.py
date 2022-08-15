from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def get_sample_group_model():
    try:
        return django_apps.get_model(
            settings.OMICS_SAMPLE_GROUP_MODEL,
            require_ready=False,
        )
    except ValueError:
        raise ImproperlyConfigured(
            "OMICS_SAMPLE_GROUP_MODEL must be of the form "
            "'app_label.model_name'"
        )
    except LookupError:
        raise ImproperlyConfigured(
            f'OMICS_SAMPLE_GROUP_MODEL refers to model '
            f'{settings.OMICS_SAMPLE_GROUP_MODEL} that has not been installed'
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
