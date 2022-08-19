from pandas import isna, Timestamp

from django.conf import settings

from mibios_umrad.manager import InputFileError, Loader
from mibios_umrad.utils import ExcelSpec, atomic_dry


class DatasetLoader(Loader):
    empty_values = ['NA', 'Not Listed', 'NF']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Amplicon_Datasets.xlsx'

    def ensure_id(self, value, row):
        """ skip rows without some id """
        for i in ['StudyID', 'NCBI_BioProject', 'JGI_Project_ID', 'GOLD_ID']:
            if row[i]:
                break
        else:
            return self.spec.SKIP_ROW

        return value

    def get_reference_ids(self, value, row):
        if value is None or value == '':
            return self.spec.IGNORE_COLUMN

        Reference = self.model._meta.get_field('reference').related_model
        id_lookups = Reference.get_accession_lookups()

        try:
            ref = Reference.objects.get(short_reference=value)
        except Reference.DoesNotExist as e:
            msg = f'unknown reference: {value}'
            raise InputFileError(msg) from e
        except Reference.MultipleObjectsReturned as e:
            # FIXME: keep this for initial dev
            msg = f'reference is not unique: {value}'
            raise InputFileError(msg) from e

        return tuple((getattr(ref, i) for i in id_lookups))

    spec = ExcelSpec(
        ('StudyID', 'study_id', ensure_id),
        ('Reference', 'reference', get_reference_ids),
        ('NCBI_BioProject', 'bioproject'),
        ('JGI_Project_ID', 'jgi_project'),
        ('GOLD_ID', 'gold_id'),
        ('Other_Databases', None),
        ('Location and Sampling Scheme', 'scheme'),
        ('Material Type', 'material_type'),
        ('Water Bodies', 'water_bodies'),
        ('Primers', 'primers'),
        ('Sequencing targets', 'sequencing_target'),
        ('Sequencing Platform', 'sequencing_platform'),
        ('Size Fraction(s)', 'size_fraction'),
        ('Notes', 'note'),
        sheet_name='studies',
    )


class RefSpec(ExcelSpec):
    """ allow skipping of empty rows """
    def iterrows(self):
        for row in super().iterrows():
            if isna(row['Reference']) or row['Reference'] == '':
                # empty row or no ref
                continue
            yield row


class ReferenceLoader(Loader):
    empty_values = ['NA', 'Not Listed']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Amplicon_Datasets.xlsx'

    spec = RefSpec(
        ('Reference', 'short_reference'),
        ('Authors', 'authors'),
        ('Paper', 'title'),
        ('Abstract', 'abstract'),
        ('Key Words', 'key_words'),
        ('Journal', 'publication'),
        ('DOI', 'doi'),
        sheet_name='studies',
    )


class SampleLoader(Loader):
    """ loader for Great_Lakes_AMplicon_Datasets.xlsx """
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A']

    def get_file(self):
        return settings.GLAMR_META_ROOT / 'Great_Lakes_Amplicon_Datasets.xlsx'

    def fix_sample_id(self, value, row):
        """ Remove leading "SAMPLE_" from accession value """
        return value.removeprefix('Sample_')

    def parse_bool(self, value, row):
        if value:
            if value == 'FALSE':
                return False
            elif value == 'TRUE':
                return True
            else:
                raise InputFileError(
                    f'expected TRUE or FALSE but got: {value}'
                )
        return value

    def check_ids(self, value, row):
        """ check that we have at least some ID value """
        if value:
            return value

        # check other ID columns
        row = self.spec.row2dict(row)
        if row['biosample'] or row['sra_accession']:
            return value

        # consider row blank
        return self.spec.SKIP_ROW

    def ensure_tz(self, value, row):
        """ add missing time zone """
        # Django would, in DateTimeField.get_prep_value(), add the configured
        # TZ for naive timestamps also, BUT would spam us with WARNINGs, so
        # this here is only to avoid those warnings.
        # Pandas should give us Timestamp instances here
        if isinstance(value, Timestamp):
            if value.tz is None and settings.USE_TZ:
                value = value.tz_localize(settings.TIME_ZONE)
        return value

    spec = ExcelSpec(
        ('StudyID', 'dataset.study_id'),
        ('SampleID', None),
        ('SampleName', 'sample_name', check_ids),
        ('NCBI_BioProject', None),
        ('Biosample', 'biosample'),
        ('Accession_Number', 'sra_accession'),
        ('sample_type', 'sample_type'),
        ('amplicon_target', 'amplicon_target'),
        ('F_primer', 'fwd_primer'),
        ('R_primer', 'rev_primer'),
        ('geo_loc_name', 'geo_loc_name'),
        ('GAZ_id', 'gaz_id'),
        ('lat', 'latitude'),
        ('lon', 'longitude'),
        ('collection_date', 'collection_timestamp', ensure_tz),
        ('NOAA_Site', 'noaa_site'),
        ('env_broad_scale', 'env_broad_scale'),
        ('env_local_scale', 'env_local_scale'),
        ('env_medium', 'env_medium'),
        ('modified_or_expermental', 'modified_or_experimental', parse_bool),
        ('depth', 'depth'),
        ('depth_sediment', 'depth_sediment'),
        ('size_frac_up', 'size_frac_up'),
        ('size_frac_low', 'size_frac_low'),
        ('pH', 'ph'),
        ('temp', 'temp'),
        ('calcium', 'calcium'),
        ('potassium', 'potassium'),
        ('magnesium', 'magnesium'),
        ('ammonium', 'ammonium'),
        ('nitrate', 'nitrate'),
        ('phosphorus', 'phosphorus'),
        ('diss_oxygen', 'diss_oxygen'),
        ('conduc', 'conduc'),
        ('secci', 'secci'),
        ('turbidity', 'turbidity'),
        ('part_microcyst', 'part_microcyst'),
        ('diss_microcyst', 'diss_microcyst'),
        ('ext_phyco', 'ext_phyco'),
        ('chlorophyl', 'chlorophyl'),
        ('diss_phosp', 'diss_phosp'),
        ('soluble_react_phosp', 'soluble_react_phosp'),
        ('ammonia', 'ammonia'),
        ('Nitrate_Nitrite', 'nitrate_nitrite'),
        ('urea', 'urea'),
        ('part_org_carb', 'part_org_carb'),
        ('part_org_nitro', 'part_org_nitro'),
        ('diss_org_carb', 'diss_org_carb'),
        ('Col_DOM', 'col_dom'),
        ('suspend_part_matter', 'suspend_part_matter'),
        ('suspend_vol_solid', 'suspend_vol_solid'),
        ('Notes', 'notes'),
        sheet_name='samples',
    )

    @atomic_dry
    def load_meta(self, **kwargs):
        """ samples meta data """
        template = dict(meta_data_loaded=True)
        super().load(template=template, validate=True, **kwargs)
