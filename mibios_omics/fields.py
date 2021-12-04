from django.conf import settings

from mibios_umrad.fields import PathField


class DataPathField(PathField):
    description = 'a path under the data root directory'
    default_base = settings.OMICS_DATA_ROOT
