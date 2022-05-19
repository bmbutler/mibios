
# Convenience functions to get the swappable models.  This tries to follow how
# the auth.User model swapping works.  BUT: we define get_sample_model() etc.
# in the utils module and then import it here for convenience.  This is
# because, if we were to define those functions here in __init__ the required
# import of the settings gives us the mibios_omics'a app settings module and
# not the expected LazySettings object for some strange reason.
from . utils import get_sample_model, get_sample_group_model  # noqa: F401


class DBRouter:
    """
    Use the default db for everything django-internal and omics everything else
    """
    # django-internal apps with models
    default_db_apps = ['admin', 'auth', 'contenttypes', 'sessions']

    def db_for_read(self, model, **hints):
        if model._meta.app_label in self.default_db_apps:
            return 'default'
        else:
            return 'omics'

    def db_for_write(self, model, **hints):
        if model._meta.app_label in self.default_db_apps:
            return 'default'
        else:
            return 'omics'

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label in self.default_db_apps and db == 'default':
            return True
        elif app_label not in self.default_db_apps and db == 'omics':
            return True
        else:
            return False
