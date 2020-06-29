"""
Definitions for special datasets
"""
import re


class Dataset():
    name = None
    model = None
    fields = []
    filter = {}
    excludes = []
    missing_data = []


class Metadata(Dataset):
    name = 'metadata'
    model = 'Sequencing'
    fields = [
        ('name', 'FASTQ_ID'),
        ('sample__participant__name', 'Participant_ID'),
        ('sample__canonical', 'Sample_ID'),
        ('sample__week', 'Study_week'),
        ('sample__participant__semester', 'Semester'),
        #  ('??', 'Use_Data'),
        ('sample__participant__quantity_compliant', 'Quantity_compliant'),
        ('sample__participant__supplement__frequency', 'Frequency'),
        ('sample__participant__supplement__dose', 'Total_dose_grams'),
        ('sample__participant__supplement__composition', 'Supplement_consumed'),
        ('sample__ph', 'pH'),
        ('sample__bristol', 'Bristol'),
        ('run__serial', 'seq_serial'),
        ('run__number', 'seq_run'),
        ('note', 'drop'),
    ]
    filter = {
        'sample__week__isnull': False,
    }
    excludes = [
        {'note__name__contains': 'drop'},
    ]


class MetadataThru2019(Metadata):
    name = 'metadata_thru2019'
    filter = {
        'sample__week__isnull': False,
        'sample__participant__semester__year__lte': '2019',
    }


class SCFA_indv(Dataset):
    name = 'SCFA_indv'
    model = 'fecalsample'
    fields = [
        ('participant', 'Participant_ID'),
        ('number', 'Sample_number'),
        ('canonical', 'Sample_ID'),
        ('week', 'Study_week'),
        ('participant__semester', 'Semester'),
        # ('use_data', 'Use_Data'),
        ('participant__quantity_compliant', 'Quantity_compliant'),
        ('participant__supplement__frequency', 'Frequency'),
        ('participant__supplement__composition', 'Supplement_consumed'),
        ('final_weight', 'Final_weight'),
        ('acetate_abs', 'Acetate_mM'),
        ('acetate_rel', 'Acetate_mmol_kg'),
        ('butyrate_abs', 'Butyrate_mM'),
        ('butyrate_rel', 'Butyrate_mmol_kg'),
        ('butyrate_abs', 'Propionate_mM'),
        ('butyrate_rel', 'Propionate_mmol_kg'),
        ('note', 'SCFA_notes'),
    ]
    missing_data = [
        re.compile(r'(NA|[-])', re.IGNORECASE)
    ]


class MMPManifest(Dataset):
    name = 'MMP_manifest'
    model = 'Sequencing'
    fields = [
        ('name', 'specimen'),
        ('batch',),
        ('r1_file', 'R1'),
        ('r2_file', 'R2'),
        ('sample__participant', 'person'),
        ('sample', 'Sample_ID'),
        ('sample__participant__semester', 'semester'),
        ('plate',),
        ('snumber', 'seqlabel'),
        ('read__1_fn',),
        ('read__2_fn',),
    ]


DATASET = {}
for i in Dataset.__subclasses__():
    DATASET[i.name] = i
