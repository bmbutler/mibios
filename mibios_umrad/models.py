from collections import OrderedDict
from logging import getLogger
from itertools import zip_longest

from django.conf import settings
from django.db import models
from django.db.transaction import atomic

from mibios.models import Model

from .model_utils import fk_opt
from .utils import ProgressPrinter


log = getLogger(__name__)


class TaxName(Model):
    history = None

    RANKS = (
        (0, 'root'),
        (1, 'domain'),
        (2, 'phylum'),
        (3, 'class', 'klass'),
        (4, 'order'),
        (5, 'family'),
        (6, 'genus'),
        (7, 'species'),
        (8, 'strain'),
    )
    RANK_CHOICE = ((i[0], i[1]) for i in RANKS)

    rank = models.PositiveSmallIntegerField(choices=RANK_CHOICE)
    name = models.CharField(max_length=64)

    class Meta:
        unique_together = (('rank', 'name'),)
        verbose_name = 'taxonomic name'

    def __str__(self):
        return f'{self.get_rank_display()} {self.name}'

    @classmethod
    @atomic
    def load(cls, path=None):
        if path is None:
            path = cls.get_taxonomy_path()

        data = []
        with path.open() as f:
            pp = ProgressPrinter('taxa found')
            log.info(f'reading taxonomy: {path}')
            for line in f:
                data.append(line.strip().split('\t'))
                pp.inc()

            pp.finish()

        pp = ProgressPrinter('tax names processed')
        rankids = [i[0] for i in TaxName.RANKS[1:]]
        names = dict()
        for row in data:
            for rid, name in zip_longest(rankids, row[1:]):
                if rid is None:
                    raise RuntimeError(f'unexpectedly low ranks: {row}')
                if name is None:
                    # no strain etc
                    continue
                key = (rid, name)
                pp.inc()
                if key not in names:
                    names[key] = TaxName(rank=rid, name=name)

        pp.finish()

        log.info(f'Storing {len(names)} unique tax names to DB...')
        TaxName.objects.bulk_create(names.values())

        return data

    @classmethod
    def get_taxonomy_path(cls):
        return (settings.OMICS_DATA_ROOT / 'NCRNA' / 'RNA_CENTRAL'
                / 'TAXONOMY_DB_2021.txt')


class Taxon(Model):
    history = None

    taxid = models.PositiveIntegerField(unique=True)
    domain = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_dom_rel',
    )
    phylum = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_phy_rel',
    )
    klass = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_cls_rel',
    )
    order = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_ord_rel',
    )
    family = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_fam_rel',
    )
    genus = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_gen_rel',
    )
    species = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_sp_rel',
    )
    strain = models.ForeignKey(
        TaxName, **fk_opt,
        related_name='tax_str_rel',
    )

    class Meta:
        verbose_name_plural = 'taxa'

    def __str__(self):
        return f'{self.taxid} {self.format_lineage(self.get_lineage())}'

    def get_lineage(self):
        lineage = []
        for i in TaxName.RANKS[1:]:
            attr = i[-1]
            name = getattr(self, attr, None)
            if name is None:
                break
            else:
                lineage.append(name.name)
        return lineage

    def names(self, with_missing_ranks=True):
        """ return dict of taxnames """
        names = OrderedDict()
        for i in TaxName.RANKS[1:]:
            attr = i[-1]
            name = getattr(self, attr, None)
            if name is None and not with_missing_ranks:
                break
            else:
                names[attr] = name

        return names

    @classmethod
    def format_lineage(cls, lineage):
        """
        Format a list of str taxnames as lineage
        """
        return '|'.join(lineage)

    @classmethod
    @atomic
    def load(cls, path=None):
        data = TaxName.load(path)

        # reloading names to get the ids, depends on order the fields are
        # declared
        names = {
            (rank, name): pk for pk, rank, name
            in TaxName.objects.values_list().iterator()
        }

        pp = ProgressPrinter('taxa processed')
        objs = []
        # ranks: get pairs of rank id and rank field attribute name
        ranks = [(i[0], i[-1]) for i in TaxName.RANKS[1:]]
        for row in data:
            kwargs = dict(taxid=row[0])
            for (rid, attr), name in zip_longest(ranks, row[1:]):
                # we should always have rid, attr here since we went through
                # data before, name may be None for missing low ranks
                if name is not None:
                    # assign ids directly
                    kwargs[attr + '_id'] = names[(rid, name)]

            objs.append(cls(**kwargs))
            pp.inc()

        pp.finish()
        log.info(f'Storing {len(objs)} taxa in DB...')
        cls.objects.bulk_create(objs)

    @classmethod
    def classified(cls, lineage):
        """ remove unclassified tail of a lineage """
        ranks = [i[1].upper() for i in cls.RANK_CHOICE[1:]]
        ret = lineage[:1]  # keep first
        for rank, name in zip(ranks, lineage[1:]):
            if name == f'UNCLASSIFIED_{ret[-1]}_{rank}':
                break
            ret.append(name)

        return ret
