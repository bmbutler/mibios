from django import template
from django.utils.html import format_html, format_html_join
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
    return format_html(
        '<ul>\n{}\n</ul>\n',
        format_html_join(
            '\n  ',
            '<li>{}: {}</li>',
            ((k, object_to_html(v)) for k, v in value.items())
        )
    )

