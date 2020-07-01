"""
Definitions for special datasets
"""
from pathlib import Path
import re


class UserDataError(Exception):
    pass


class Dataset():
    name = None
    model = None
    fields = []
    filter = {}
    excludes = []
    missing_data = []

    # instance container for subclass singletons
    __instance = {}

    def __new__(cls):
        """
        Create per-subclass singleton instance
        """
        if cls.__name__ not in cls.__instance:
            cls.__instance[cls.__name__] = super().__new__(cls)
        return cls.__instance[cls.__name__]


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
        # ('name', 'specimen'),
        # ('batch',),
        ('r1_file', 'R1'),
        ('r2_file', 'R2'),
        ('sample__participant', 'person'),
        ('sample__canonical', 'Sample_ID'),
        ('sample__participant__semester', 'semester'),
        ('plate', 'plate'),
        ('snumber', 'seqlabel'),
        # ('', read__1_fn'),
        # ('', read__2_fn'),
    ]

    name_extra_pat = re.compile(r'_L[0-9].*')
    plate_pat = re.compile(r'^P([0-9])-([A-Z][0-9]+)$')
    snum_plus_pat = re.compile(r'(_S[0-9]+).*$')

    def parse_plate(self, txt):
        """
        Extract plate and position
        """
        m = self.plate_pat.match(txt)
        if m is None:
            raise UserDataError(
                'Failed matching patter: {}'.format(self.plate_pat.pattern)
            )
        plate, position = m.groups()
        plate = int(plate)
        return dict(plate=plate, plate_position=position)

    def parse_r1_file(self, txt):
        """ extract record name and normalize path """
        p = Path(txt)
        name = self.snum_plus_pat.sub(r'\1', p.stem)
        name = name.replace('-', '_')
        return {
            'name': name,
            'r1_file': str(p),
        }

    def parse_r2_file(self, txt):
        """ normalize path """
        return str(Path(txt))

    def parse_snumber(self, txt):
        """ extract number as int """
        return int(txt.lstrip('S'))


class ParticipantList(Dataset):
    name = 'participants_list'
    model = 'Participant'
    fields = [
        ('name', 'Participant_ID'),
        ('semester', 'Semester'),
        ('has_consented', 'Use_Data'),
        ('saliva_status', 'Saliva'),
        ('supplement_status', 'Dietary_Supplement'),
        ('blood_status', 'Blood'),
        ('has_consented_future', 'Use Data in Unspecified Future Research'),
        ('has_consented_contact', 'Contact for Future Study Participation'),
        ('note', 'Notes'),
    ]

    def parse_has_consented(self, txt):
        """
        Turn mostly yes/no column into bool
        """
        txt = txt.casefold()
        if not txt:
            return None
        if txt == 'yes':
            return True
        return False

    def parse_has_consented_future(self, txt):
        return self.parse_has_consented(txt)


DATASET = {}
for i in Dataset.__subclasses__():
    DATASET[i.name] = i()
