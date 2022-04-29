from .utils import getLogger
from .views import BaseMixin


log = getLogger(__name__)


_base_context = None


def base(request):
    """
    Pre-populate context for the base.html tempate

    This can be added to settings.TEMPLATE['context_processors']

    This context processor is for the benefit of using the
    django.views.default error views whose context we can't otherwise
    manipulate without writing our own error handling views.

    Our regular views in mibios.views etc should always overwrite these
    context variables with their own value.  For those the overwrite happens in
    django.template.context.make_context().
    """
    global _base_context
    if _base_context is None:
        # the base context should be static, so only build it once, intended to
        # be failsafe because this is used for error views
        try:
            _base_context = BaseMixin().get_context_data()
        except Exception as e:
            try:
                log.warning(
                    f'Failed getting base context: {e.__class__.__name__}: {e}'
                )
            except Exception:
                pass
            _base_context = {}

    return _base_context
