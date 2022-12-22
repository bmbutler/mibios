from datetime import datetime
import re

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime

from mibios.umrad.manager import InputFileError, Loader
from mibios.umrad.model_utils import delete_all_objects_quickly
from mibios.umrad.utils import CSV_Spec, atomic_dry


class MetaDataLoader(Loader):
    default_load_kwargs = dict(
        validate=True,
        bulk=False,
        update=True,
        diff=True,
    )


class DatasetLoader(MetaDataLoader):
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx - studies_datasets.tsv'

    def ensure_id(self, value, obj):
        """ Pre-processor to skip rows without dataset id """
        if not value:
            return self.spec.SKIP_ROW

        return value

    def get_reference_ids(self, value, obj):
        # FIXME: is unused, can this be removed?
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

    spec = CSV_Spec(
        ('dataset', 'dataset_id', ensure_id),
        # ('Primary_pub', 'reference', get_reference_ids),
        ('Primary_pub', 'reference.reference_id'),
        ('primary_pub_title', None),
        ('NCBI_BioProject', 'bioproject'),
        ('JGI_Project_ID', 'jgi_project'),
        ('GOLD_ID', 'gold_id'),
        ('MG-RAST_study', None),  # TODO: add
        ('Location and Sampling Scheme', 'scheme'),
        ('Material Type', 'material_type'),
        ('Water Bodies', 'water_bodies'),
        ('Primers', 'primers'),
        ('Sequencing targets', 'sequencing_target'),
        ('Sequencing Platform', 'sequencing_platform'),
        ('Size Fraction(s)', 'size_fraction'),
        ('Notes', 'note'),
    )


class ReferenceLoader(MetaDataLoader):
    empty_values = ['NA', 'Not Listed']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Omics_Datasets.xlsx - papers.tsv'

    def fix_doi(self, value, obj):
        """ Pre-processor to fix issue with some DOIs """
        if value is not None and 'doi-org.proxy.lib.umich.edu' in value:
            # fix, don't require umich weblogin to follow these links
            value = value.replace('doi-org.proxy.lib.umich.edu', 'doi.org')
        return value

    def check_skip(self, value, obj):
        """ Pre-processor to determine if row needs to be skipped """
        if value == 'paper_17':
            return self.spec.SKIP_ROW

        if value == 'Koeppel et al. 2022':
            return self.spec.SKIP_ROW

        if not value:
            return self.spec.SKIP_ROW

        return value

    spec = CSV_Spec(
        ('PaperID', 'reference_id', check_skip),
        ('Reference', 'short_reference', check_skip),
        ('Authors', 'authors'),
        ('Title', 'title'),
        ('Abstract', 'abstract'),
        ('Key Words', 'key_words'),
        ('Journal', 'publication'),
        ('DOI', 'doi', fix_doi),
        ('Associated_datasets', None),  # TODO: handle this
    )


class SampleLoader(MetaDataLoader):
    """ loader for Great_Lakes_Omics_Datasets.xlsx """
    empty_values = ['NA', 'Not Listed', 'NF', '#N/A']

    def get_file(self):
        return settings.GLAMR_META_ROOT / 'Great_Lakes_Omics_Datasets.xlsx - samples.tsv'  # noqa:E501

    def fix_sample_id(self, value, obj):
        """ Remove leading "SAMPLE_" from accession value """
        return value.removeprefix('Sample_')

    def parse_bool(self, value, obj):
        """ Pre-processor to parse booleans """
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

    def check_ids(self, value, obj):
        """ Pre-processor to check that we have at least some ID value """
        if value:
            return value

        # check other ID columns
        if self.get_current_value('biosample'):
            return value
        if self.get_current_value('sra_accession'):
            return value

        # consider row blank
        return self.spec.SKIP_ROW

    # re for yyyy or yyyy-mm (year or year/month only) timestamp formats
    partial_date_pat = re.compile(r'^([0-9]{4})(?:-([0-9]{2})?)$')

    def process_timestamp(self, value, obj):
        """
        Pre-processor for the collection time stamp

        1. Partial dates: years only get saved as Jan 1st and year-month gets
           the first of month; this is indicated by the collection_ts_partial
           field, which is set here.
        2. Add timezone to timestamps if needed (and hoping the configued TZ is
           appropriate).  This avoids spamming of warnings from the
           DateTimeField's to_python() method in some cases.
        """
        if value is None:
            return None

        old_value = value
        del value

        try:
            value = parse_datetime(old_value)
        except ValueError as e:
            # invalid date/time
            raise InputFileError(e) from e

        if value is None:
            # failed parsing as datetime, try date
            try:
                value = parse_date(old_value)
            except ValueError as e:
                # invalid date
                raise InputFileError(e) from e

            if value is not None:
                # add fake time (midnight)
                value = datetime(value.year, value.month, value.day)
                collection_ts_partial = self.model.DATE_ONLY
        else:
            # got a complete timestamp
            collection_ts_partial = self.model.FULL_TIMESTAMP

        if value is None:
            # failed parsing as ISO 8601, try for year-month and year only
            # which MIMARKS/MIXS allows
            m = self.partial_date_pat.match(old_value)

            if m is not None:
                year, month = m.groups()
                year = int(year)
                if month is None:
                    month = 1
                    collection_ts_partial = self.model.YEAR_ONLY
                else:
                    month = int(month)
                    collection_ts_partial = self.model.MONTH_ONLY

                try:
                    value = datetime(year, month, 1)
                except ValueError as e:
                    # invalid year or month
                    raise InputFileError('failed parsing timestamp', e) from e

        # value should now be a datetime instance
        if value.tzinfo is None:
            default_timezone = timezone.get_default_timezone()
            value = timezone.make_aware(value, default_timezone)

        obj.collection_ts_partial = collection_ts_partial
        return value

    def parse_human_int(self, value, obj):
        """
        Pre-processor to allow use of commas to separate thousands
        """
        if value:
            return value.replace(',', '')
        else:
            return value

    spec = CSV_Spec(
        ('SampleID', 'sample_id'),  # B
        ('SampleName', 'sample_name', check_ids),  # C
        ('StudyID', 'dataset.dataset_id'),  # A
        ('ProjectID', 'project_id'),  # D
        ('Biosample', 'biosample'),  # E
        ('Accession_Number', 'sra_accession'),  # F
        ('JGI_study', None),  # TODO
        ('JGI_biosample', None),  # TODO
        ('sample_type', 'sample_type'),  # G
        ('has_paired_data', 'has_paired_data', parse_bool),  # H
        ('amplicon_target', 'amplicon_target'),  # I
        ('F_primer', 'fwd_primer'),  # J
        ('R_primer', 'rev_primer'),  # K
        ('geo_loc_name', 'geo_loc_name'),  # L
        ('GAZ_id', 'gaz_id'),  # M
        ('lat', 'latitude'),  # N
        ('lon', 'longitude'),  # O
        ('collection_date', 'collection_timestamp', process_timestamp),  # P
        (CSV_Spec.CALC_VALUE, 'collection_ts_partial', None),  # P
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
        ('Microcystis_count', 'microcystis_count', parse_human_int),  # BC
        ('Planktothris_Count', 'planktothris_count', parse_human_int),  # BD
        ('Anabaena-D_count', 'anabaena_d_count', parse_human_int),  # BE
        ('Cylindrospermopsis_count', 'cylindrospermopsis_count'),  # BF
        ('ice_cover', None),  # TODO
        ('Notes', 'notes'),  # BG
    )

    @atomic_dry
    def load_meta(self, **kwargs):
        """ samples meta data """
        template = dict(meta_data_loaded=True)
        return self.load(template=template, **kwargs)


class SearchTermManager(Loader):
    def reindex(self):
        delete_all_objects_quickly(self.model)
        from .search_utils import spellfix_models as models, update_spellfix
        for i in models:
            self._index_model(i)
        update_spellfix()

    def _index_model(self, model):
        """
        update the search index
        """
        if model._meta.model_name == 'compoundname':
            abund_lookup = 'compoundrecord__abundance'
        elif model._meta.model_name == 'functionname':
            abund_lookup = 'funcrefdbentry__abundance'
        else:
            try:
                model._meta.get_field('abundance')
            except FieldDoesNotExist:
                abund_lookup = None
            else:
                abund_lookup = 'abundance'
        print(f'Collecting search terms for {model._meta.verbose_name}... ',
              end='', flush=True)
        if abund_lookup:
            # get PKs of objects with hits/abundance
            f = {abund_lookup: None}
            whits = model.objects.exclude(**f).values_list('pk', flat=True)
            whits = set((i for i in whits.iterator()))
            print(f'with hits: {len(whits)} / total: ', end='', flush=True)
        else:
            whits = set()

        fname = model.get_search_field().name
        qs = model.objects.all()
        print(f'{qs.count()} [OK]')
        max_length = self.model._meta.get_field('term').max_length
        objs = (
            self.model(
                term=getattr(i, fname)[:max_length],
                has_hit=(i.pk in whits),
                content_object=i,
            )
            for i in qs.iterator()
        )
        self.bulk_create(objs)
