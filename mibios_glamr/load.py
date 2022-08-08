from pandas import isna

from django.conf import settings

from mibios_umrad.manager import InputFileError, Loader
from mibios_umrad.utils import ExcelSpec, CSV_Spec, atomic_dry


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
        ('StudyID', 'accession', ensure_id),
        ('Reference', 'reference', get_reference_ids),
        ('NCBI_BioProject', 'bioproject'),
        ('JGI_Project_ID', 'jgi_project'),
        ('GOLD_ID', 'gold_id'),
        ('Other_Databases', None),
        ('Location and Sampling Scheme', 'scheme'),
        ('Material Type', 'material_type'),
        ('Water Bodies', 'water_bodies'),
        ('Primers', 'primers'),
        ('Gene target', 'gene_target'),
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

    spec = CSV_Spec()

    def get_file(self):
        return settings.GLAMR_META_ROOT / '2014_metaG_metadata.tsv'

    def fix_sample_id(self, value, row):
        """ Remove leading "SAMPLE_" from accession value """
        return value.removeprefix('Sample_')

    @atomic_dry
    def load_erie2014(self, update=False):
        fnames = [
            'sample_id',
            'site',
            'fraction',
            'sample_name',
            'date',
            'station_depth',
            'sample_depth',
            'sample_depth_category',
            'local_time',
            'latitude',
            'longitude',
            'wind_speed',
            'wave_height',
            'sky',
            'secchi_depth',
            'sample_temperature',
            'ctd_temperature',
            'ctd_specific_conductivity',
            'ctd_beam_attenuation',
            'ctd_tramission',
            'ctd_dissolved_oxygen',
            'ctd_radiation',
            'turbidity',
            'particulate_microcystin',
            'dissolved_microcystin',
            'extracted_phycocyanin',
            'extracted_chlorophyll_a',
            'phosphorus',
            'dissolved_phosphorus',
            'soluble_reactive_phosphorus',
            'ammonia',
            'nitrate_nitrite',
            'urea',
            'organic_carbon',
            'organic_nitrogen',
            'dissolved_organic_carbon',
            'absorbance',
            'suspended_solids',
            'Volatile_suspended_solids']

        # get column headers from verbose names!
        specs = []
        for i in fnames:
            # column spec format is:
            # (<colum header>, <field name>, [conversion function])
            if i == 'sample_id':
                specs.append(('accession', i, 'fix_sample_id'))
            else:
                field = self.model._meta.get_field(i)
                specs.append((field.verbose_name, i))
        self.empty_values = ['NA']
        self.spec.setup(loader=self, column_specs=specs)

        # FIXME: need some proper source for dataset
        Dataset = self.model._meta.get_field('group').related_model
        dset, new = Dataset.objects.get_or_create(
            short_name='2014 Metagenomes',
        )
        template = dict(group=dset)
        super().load(update=update, template=template)
