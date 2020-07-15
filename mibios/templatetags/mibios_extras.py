from django import template
from django.utils.safestring import mark_safe


register = template.Library()


@register.filter
def prettyformat(value):
    return mark_safe(object_to_html(value))


def object_to_html(value):
    if isinstance(value, dict):
        return dict_to_html(value)
    return str(value)


def dict_to_html(value):
    return '<ul>{}\n</ul>\n'.format(
        '\n  '.join([
            '<li>{}: {}</li>'.format(k, object_to_html(v))
            for k, v in value.items()
        ])
    )

