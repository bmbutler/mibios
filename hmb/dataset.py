"""
Definitions for special datasets
"""
import re


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
DATASET['meta_thru2019']['filter']['sample__participant__semester__year__lte']\
    = '2019'

DATASET['SCFA_indv'] = {
    'model': 'fecalsample',
    'fields': [
        ('participant', 'Participant_ID'),
        ('number', 'Sample_number'),
        ('canonical', 'Sample_ID'),
        ('week', 'Study_week'),
        ('participant__semester', 'Semester'),
        # ('use_data', 'Use_Data'),
        ('participant__quantity_compliant', 'Quantity_compliant'),
        ('participant__diet__frequency', 'Frequency'),
        ('participant__diet__supplement', 'Supplement_consumed'),
        ('final_weight', 'Final_weight'),
        ('acetate_abs', 'Acetate_mM'),
        ('acetate_rel', 'Acetate_mmol_kg'),
        ('butyrate_abs', 'Butyrate_mM'),
        ('butyrate_rel', 'Butyrate_mmol_kg'),
        ('butyrate_abs', 'Propionate_mM'),
        ('butyrate_rel', 'Propionate_mmol_kg'),
        ('note', 'SCFA_notes'),
    ],
    'filter': {},
    'missing_data': [
        re.compile(r'(NA|[-])', re.IGNORECASE)
    ],
}
