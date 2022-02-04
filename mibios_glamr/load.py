from itertools import groupby

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models.manager import BaseManager
from django.db.transaction import atomic, set_rollback


class DatasetLoader(BaseManager):
    empty_values = ['NA', 'Not Listed']

    def get_file(self):
        return settings.GLAMR_META_ROOT\
            / 'Great_Lakes_Amplicon_Datasets.xlsx - Database.tsv'

    cols = (
        ('Reference', 'short_reference'),
        ('Authors', 'authors'),
        ('Paper', 'title'),
        ('Abstract', 'abstract'),
        ('Key Words', 'key_words'),
        ('Journal', 'publication'),
        ('DOI', 'doi'),
        # --- divides reference and data set columns ---
        ('Accession  Number', 'accession'),  # two spaces !!!
        ('Database', 'accession_db'),
        ('Location and Sampling Scheme', 'scheme'),
        ('Material Type', 'material_type'),
        ('Water Bodies', 'water_bodies'),
        ('Primers', 'primers'),
        ('Gene target', 'gene_target'),
        ('Sequencing Platform', 'sequencing_platform'),
        ('Size Fraction(s)', 'size_fraction'),
        ('Notes', 'note'),
    )

    @atomic
    def load(self, dry_run=False):
        refs = self.load_references(dry_run=False)
        with self.get_file().open() as f:
            head = f.readline().rstrip('\n').split('\t')
            if head != [i for i, _ in self.cols]:
                raise RuntimeError(f'unexpected header in {f.name}: {head}')

            field_names = [i for _, i in self.cols[7:]]
            rows = (line.rstrip('\n').split('\t')[7:] for line in f)
            for i, row in enumerate(rows):
                kw = dict((
                    (i, j) for i, j in zip(field_names, row)
                    if j and j not in self.empty_values
                ))
                kw['reference'] = refs[i]
                obj = self.model(**kw)
                try:
                    obj.full_clean()
                except ValidationError:
                    print(f'invalid at line {i + 2} -- {vars(obj)=}')
                    raise
                obj.save()

        set_rollback(dry_run)

    @atomic
    def load_references(self, dry_run=False):
        cols = self.cols[:7]  # pick reference columns
        ref_model = self.model._meta.get_field('reference').related_model
        refs = {}  # return value, maps row number to reference
        with self.get_file().open() as f:
            head = f.readline().rstrip('\n').split('\t')[:7]
            if head != [i for i, _ in cols]:
                raise RuntimeError(f'unexpected header in {f.name}: {head}')
            rows = (line.rstrip('\n').split('\t') for line in f)
            rows = enumerate(rows)

            def getrefcols(item):
                """ the sort/group key, get reference columns """
                # FIXME: only picks row up to Journal but not DOI, need to find
                # out why we have multiple DOIs per reference in source file
                return item[1][:6]
            for row, grp in groupby(sorted(rows, key=getrefcols), getrefcols):
                grp = list(grp)
                kwargs = {
                    field_name: value
                    for field_name, value in zip([i for _, i in cols[:6]], row)
                }
                doi = grp[0][1][6]  # use 1st doi in group
                if 'doi-org.proxy.lib.umich.edu' in doi:
                    # fix, don't require umich weblogin
                    doi = doi.replace('doi-org.proxy.lib.umich.edu', 'doi.org')
                kwargs['doi'] = doi
                obj = ref_model(**kwargs)
                try:
                    obj.full_clean()
                except ValidationError:
                    print(f'invalid: {vars(obj)=}')
                    raise
                obj.save()
                for (i, _) in grp:
                    refs[i] = obj

        set_rollback(dry_run)
        return refs
