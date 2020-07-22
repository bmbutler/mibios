"""
Model-related utilities

Separate module to avoid circular imports issues with dataset
"""
import sys

from .models import ChangeRecord
from .dataset import registry


def erase_all_data(verbose=False):
    """
    Delete all data
    """
    if verbose:
        print('Erasing all data...', file=sys.stderr)
    for m in registry.get_models() + [ChangeRecord]:
        m.objects.all().delete()


def show_stats():
    """
    print db stats
    """
    for m in registry.get_models() + [ChangeRecord]:
        print('{}: {}'.format(m._meta.label, m.objects.count()))
