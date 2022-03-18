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
