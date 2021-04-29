from django import template
from django.utils.html import format_html, format_html_join
from django.utils.safestring import mark_safe


register = template.Library()


@register.filter
def prettyformat(value):
    return mark_safe(object_to_html(value))


@register.filter
def prettychanges(value):
    return mark_safe(changes_to_html(value))


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


def changes_to_html(value):
    """
    Like dict_to_html but for dicts from ChangeRecord.diff()
    """
    if not isinstance(value, dict):
        return object_to_html(value)

    templ = '<li>{}: {} -> {}</li>'
    NEW_FIELD = mark_safe('&lang;new field&rang;')
    BLANK = mark_safe('&lang;blank&rang;')
    NONE = mark_safe('&lang;none&rang;')

    row_args = []
    for k, (*old, new) in value.items():
        if old:
            old = old[0]
            if old == '':
                old = BLANK
            elif old is None:
                old = NONE
        else:
            # only single item in value value
            old = NEW_FIELD
        if new == '':
            new = BLANK
        elif new is None:
            new = NONE
        row_args.append((k, old, new))

    templ = templ.replace(' ', '&nbsp;').replace('->', '&rarr;')
    return format_html(
        '<ul>\n{}\n</ul>\n',
        format_html_join('\n  ', templ, row_args)
    )
