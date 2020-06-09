"""
Definitions for special datasets
"""
__dir__ = ('x', 'a')
DATASET = {}

DATASET['meta_all'] = {
    'model': 'Sequencing',
    'fields': [
        ('name', 'FASTQ_ID'),
        ('sample__participant__name', 'Participant_ID'),
        ('sample', 'Sample_ID'),
        ('sample__week', 'Study_week'),
        ('sample__participant__semester', 'Semester'),
        ('sample__participant__quantity_compliant', 'Quantity_compliant'),
        ('sample__participant__diet__frequency', ),
        ('sample__participant__diet__dose', ),
        ('sample__participant__diet__supplement', ),
        ('sample__ph', ),
        ('sample__bristol', ),
        ('run__serial', ),
        ('run__number', ),
    ],
    'filter': {
        'sample__week__isnull': False,
    },
}


DATASET['meta_thru2019'] = DATASET['meta_all'].copy()
DATASET['meta_thru2019']['filter']['sample__participant__semester__year__lte'] = '2019'
