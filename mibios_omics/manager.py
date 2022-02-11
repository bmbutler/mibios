from logging import getLogger

from django.conf import settings
from django.db.transaction import atomic, set_rollback

from mibios_umrad.model_utils import BaseManager

log = getLogger(__name__)


class SampleManager(BaseManager):
    """ Manager for the Sample """
    def get_file(self):
        return settings.OMICS_DATA_ROOT / 'sample_list.txt'

    @atomic
    def sync(self, source_file=None, dry_run=False, **kwargs):
        if source_file is None:
            source_file = self.get_file()

        with open(source_file) as f:
            seen = []
            for line in f:
                obj, isnew = self.get_or_create(
                    accession=line.strip()
                )
                seen.append(obj.pk)
                if isnew:
                    log.info(f'new sample: {obj}')

        not_in_src = self.exclude(pk__in=seen)
        if not_in_src.exists():
            log.warning(f'Have {not_in_src.count()} extra samples in DB not '
                        f'found in {source_file}')
        set_rollback(dry_run)

    def status(self):
        if not self.exists():
            print('no samples in database yet')
            return

        print(' ' * 10, 'contigs', 'bins', 'checkm', 'genes', sep='\t')
        for i in self.all():
            print(
                f'{i}:',
                'OK' if i.contigs_ok else '',
                'OK' if i.binning_ok else '',
                'OK' if i.checkm_ok else '',
                'OK' if i.genes_ok else '',
                sep='\t'
            )

    def status_long(self):
        if not self.exists():
            print('no samples in database yet')
            return

        print(' ' * 10, 'cont cl', 'MAX', 'MET93', 'MET97', 'MET99', 'genes',
              sep='\t')
        for i in self.all():
            print(
                f'{i}:',
                i.contigcluster_set.count(),
                i.binmax_set.count(),
                i.binmet93_set.count(),
                i.binmet97_set.count(),
                i.binmet99_set.count(),
                i.gene_set.count(),
                sep='\t'
            )
