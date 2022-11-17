
# Convenience functions to get the swappable models.  This tries to follow how
# the auth.User model swapping works.  BUT: we define get_sample_model() etc.
# in the utils module and then import it here for convenience.  This is
# because, if we were to define those functions here in __init__ the required
# import of the settings gives us the mibios_omics'a app settings module and
# not the expected LazySettings object for some strange reason.
from . utils import get_sample_model, get_dataset_model  # noqa: F401
