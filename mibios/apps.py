from django import apps


class AppConfig(apps.AppConfig):
    name = 'mibios'
    label = 'microbiome_data'
    verbose_name = 'microbiome database'
