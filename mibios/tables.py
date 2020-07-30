from django.urls import reverse
from django.utils.html import format_html

import django_tables2 as tables

from .models import ChangeRecord


NONE_LOOKUP = 'NULL'


class CountColumn(tables.Column):
    def __init__(self, related_object, view=None, **kwargs):
        url = reverse(
            'queryset_index',
            kwargs=dict(dataset=related_object.name)
        )
        # our (this column's) name
        our_name = related_object.remote_field.name

        if 'linkify' not in kwargs:
            def linkify(record):
                f = {our_name: record.natural}
                query = view.build_query_string(filter=f)
                return url + query

            kwargs.update(linkify=linkify)

        # prepare URL for footer
        f = {our_name + '__' + k: v for k, v in view.filter.items()}

        elist = []
        for i in view.excludes:
            e = {our_name + '__' + k: v for k, v in i.items()}
            if e:
                elist.append(e)
        # if there is a filter selecting for us, then skip exclusion of missing
        # data:
        for i in f:
            if i.startswith(our_name):
                break
        else:
            elist.append({our_name: NONE_LOOKUP})

        q = view.build_query_string(filter=f, excludes=elist,
                                    negate=view.negate)
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


class ManyToManyColumn(tables.ManyToManyColumn):
    def __init__(self, *args, **kwargs):
        if 'default' not in kwargs:
            kwargs['default'] = ''

        super().__init__(*args, **kwargs)


class Table(tables.Table):
    pass


class HistoryTable(tables.Table):
    class Meta:
        model = ChangeRecord
        fields = (
            'timestamp', 'is_created', 'is_deleted', 'user', 'file.file', 'line',
            'command_line', 'fields',
        )


class DeletedHistoryTable(tables.Table):
    record_natural = tables.Column(
        verbose_name='record name',
        linkify=(
            'record_history',
            {
                'dataset': tables.A('record_type.model'),
                'natural': tables.A('record_natural') or tables.A('record_pk'),
            }
        )
    )

    class Meta:
        model = ChangeRecord
        fields = ('timestamp', 'user', 'record_natural',)
