from pandas import isna, Timestamp

from django.conf import settings

from mibios_umrad.manager import InputFileError, Loader
from mibios_umrad.utils import ExcelSpec, atomic_dry


class DatasetLoader(Loader):
    empty_values = ['NA', 'Not Listed', 'NF']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx'

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
        ('StudyID', 'dataset_id', ensure_id),
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
            / 'Great_Lakes_Omics_Datasets.xlsx'

    def fix_doi(self, value, record):
        if value is not None and 'doi-org.proxy.lib.umich.edu' in value:
            # fix, don't require umich weblogin to follow these links
            value = value.replace('doi-org.proxy.lib.umich.edu', 'doi.org')
        return value

    spec = RefSpec(
        ('Reference', 'short_reference'),
        ('Authors', 'authors'),
        ('Paper', 'title'),
        ('Abstract', 'abstract'),
        ('Key Words', 'key_words'),
        ('Journal', 'publication'),
        ('DOI', 'doi', fix_doi),
        sheet_name='studies',
    )


class SampleLoader(Loader):
    """ loader for Great_Lakes_AMplicon_Datasets.xlsx """
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A']

    def get_file(self):
        return settings.GLAMR_META_ROOT / 'Great_Lakes_Omics_Datasets.xlsx'

    def fix_sample_id(self, value, row):
        """ Remove leading "SAMPLE_" from accession value """
        return value.removeprefix('Sample_')

    def parse_bool(self, value, row):
        # Only parse str values.  The pandas reader may give us booleans
        # already for some reason (for the modified_or_experimental but not the
        # has_paired data) ?!?
        if isinstance(value, str) and value:
            if value.casefold() == 'false':
                return False
            elif value.casefold() == 'true':
                return True
            else:
                raise InputFileError(
                    f'expected TRUE or FALSE (any case) but got: {value}'
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
        elif isna(value):
            # blank fields get type NaTType
            value = None
        return value

    spec = ExcelSpec(
        ('StudyID', 'dataset.dataset_id'),  # A
        ('SampleID', 'sample_id'),  # B
        ('SampleName', 'sample_name', check_ids),  # C
        ('ProjectID', 'project_id'),  # D
        ('Biosample', 'biosample'),  # E
        ('Accession_Number', 'sra_accession'),  # F
        ('sample_type', 'sample_type'),  # G
        ('has_paired_data', 'has_paired_data', parse_bool),  # H
        ('amplicon_target', 'amplicon_target'),  # I
        ('F_primer', 'fwd_primer'),  # J
        ('R_primer', 'rev_primer'),  # K
        ('geo_loc_name', 'geo_loc_name'),  # L
        ('GAZ_id', 'gaz_id'),  # M
        ('lat', 'latitude'),  # N
        ('lon', 'longitude'),  # O
        ('collection_date', 'collection_timestamp', ensure_tz),  # P
        ('NOAA_Site', 'noaa_site'),  # Q
        ('env_broad_scale', 'env_broad_scale'),  # R
        ('env_local_scale', 'env_local_scale'),  # S
        ('env_medium', 'env_medium'),  # T
        ('modified_or_experimental', 'modified_or_experimental', parse_bool),
        ('depth', 'depth'),  # V
        ('depth_sediment', 'depth_sediment'),  # W
        ('size_frac_up', 'size_frac_up'),  # X
        ('size_frac_low', 'size_frac_low'),  # Y
        ('pH', 'ph'),  # Z
        ('temp', 'temp'),  # AA
        ('calcium', 'calcium'),  # AB
        ('potassium', 'potassium'),  # AC
        ('magnesium', 'magnesium'),  # AD
        ('ammonium', 'ammonium'),  # AE
        ('nitrate', 'nitrate'),  # AF
        ('phosphorus', 'phosphorus'),  # AG
        ('diss_oxygen', 'diss_oxygen'),  # AH
        ('conduc', 'conduc'),  # AI
        ('secci', 'secci'),  # AJ
        ('turbidity', 'turbidity'),  # AK
        ('part_microcyst', 'part_microcyst'),  # AL
        ('diss_microcyst', 'diss_microcyst'),  # AM
        ('ext_phyco', 'ext_phyco'),  # AN
        ('ext_microcyst', 'ext_microcyst'),  # AO
        ('ext_Anatox', 'ext_anatox'),  # AP
        ('chlorophyl', 'chlorophyl'),  # AQ
        ('diss_phosp', 'diss_phosp'),  # AR
        ('soluble_react_phosp', 'soluble_react_phosp'),  # AS
        ('ammonia', 'ammonia'),  # AT
        ('Nitrate_Nitrite', 'nitrate_nitrite'),  # AU
        ('urea', 'urea'),  # AV
        ('part_org_carb', 'part_org_carb'),  # AW
        ('part_org_nitro', 'part_org_nitro'),  # AX
        ('diss_org_carb', 'diss_org_carb'),  # AY
        ('Col_DOM', 'col_dom'),  # AZ
        ('suspend_part_matter', 'suspend_part_matter'),  # BA
        ('suspend_vol_solid', 'suspend_vol_solid'),  # BB
        ('Microcystis_count', 'microcystis_count'),  # BC
        ('Planktothris_Count', 'planktothris_count'),  # BD
        ('Anabaena-D_count', 'anabaena_d_count'),  # BE
        ('Cylindrospermopsis_count', 'cylindrospermopsis_count'),  # BF
        ('Notes', 'notes'),  # BG
        sheet_name='samples',
    )

    @atomic_dry
    def load_meta(self, validate=True, **kwargs):
        """ samples meta data """
        template = dict(meta_data_loaded=True)
        super().load(template=template, validate=validate, **kwargs)
