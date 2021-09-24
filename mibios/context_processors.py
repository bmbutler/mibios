from .utils import getLogger
from .views import BaseMixin


log = getLogger(__name__)


_base_context = {}


def base(request):
    """
    Pre-populate context for the base.html tempate

    This can be added to settings.TEMPLATE['context_processors']

    This context processor is for the benefit of using the
    django.views.default error views whose context we can't otherwise
    manipulate without writing our own error handling views.

    Our regular views in mibios.views etc should always overwrite these
    context variables with their own value.
    """
    global _base_context
    if not _base_context:
        # the base context should be static, so only build it once, intended to
        # be failsafe because this is used for error views
        try:
            _base_context = BaseMixin().get_context_data()
            _base_context['view'].request = request
        except Exception as e:
            try:
                log.warning(
                    f'Failed getting base context: {e.__class__.__name__}: {e}'
                )
            except Exception:
                pass

    return _base_context
