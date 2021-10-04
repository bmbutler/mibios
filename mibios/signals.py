from django.core.cache import caches
from django.db.models.signals import post_save
from django.dispatch import receiver


from .utils import getLogger


log = getLogger('mibios')


@receiver(post_save)
def clear_cache_on_save(sender, **kwargs):
    """
    Empty the cache when something gets saved
    """
    c = caches['default']
    if not hasattr(c, '_cache'):
        # dummy cache
        return

    if c._cache:
        log.debug(f'cache: clearing all {len(c._cache.keys())} entries')
        c.clear()
