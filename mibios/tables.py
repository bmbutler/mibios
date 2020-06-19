from django.urls import reverse
from django.utils.html import format_html

import django_tables2 as tables

from .models import Sequencing


NONE_LOOKUP = 'NULL'


class CountColumn(tables.Column):
    def __init__(self, related_object, view=None, **kwargs):
        url = reverse(
            'queryset_index',
            kwargs=dict(dataset=related_object.name)
        )
        # our name: not always the same as our model/dataset_name
        if 'linkify' not in kwargs:
            def linkify(record):
                f = {record._meta.model_name + '__pk': record.pk}
                query = view.get_query_string(ignore_original=True, filter=f)
                return url + query

            kwargs.update(linkify=linkify)

        # prepare URL for footer
        our_name = related_object.remote_field.name
        f = {our_name + '__' + k: v for k, v in view.filter.items()}
        e = {our_name + '__' + k: v for k, v in view.exclude.items()}
        e[our_name] = NONE_LOOKUP
        q = view.get_query_string(ignore_original=True, filter=f, exclude=e)
        self.footer_url = url + q

        super().__init__(self, **kwargs)
        # verb name can be set after __init__, setting explicitly before
        # interferes with the automatic column class selection
        self.verbose_name = related_object.name + ' count'

    def render_footer(self, bound_column, table):
        total = 0
        for row in table.data:
            total += bound_column.accessor.resolve(row)
        return format_html('all: <a href={}>{}</a>', self.footer_url, total)


class SequencingTable(tables.Table):
    class Meta:
        model = Sequencing
        template_name = 'django_tables2/bootstrap.html'
        fields = (
            'name', 'sample.participant', 'sample', 'sample.week',
            'sample.participant.semester', 'batch', 'note',
        )


class Table(tables.Table):
    name = tables.Column(linkify=True)
    id = tables.Column(linkify=True)
