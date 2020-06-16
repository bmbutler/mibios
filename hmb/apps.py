from django import apps


class AppConfig(apps.AppConfig):
    name = 'hmb'
    label = 'microbiome_data'
    verbose_name = 'microbiome database'
