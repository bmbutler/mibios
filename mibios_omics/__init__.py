from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def get_sample_group_model():
    try:
        return django_apps.get_model(
            settings.SAMPLE_GROUP_MODEL,
            require_ready=False,
        )
    except ValueError:
        raise ImproperlyConfigured(
            "SAMPLE_GROUP_MODEL must be of the form 'app_label.model_name'"
        )
    except LookupError:
        raise ImproperlyConfigured(
            f'SAMPLE_GROUP_MODEL refers to model {settings.SAMPLE_GROUP_MODEL}'
            f' that has not been installed'
        )
