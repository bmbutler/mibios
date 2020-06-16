from django.urls import reverse

import django_tables2 as tables

from .models import Sequencing


class CountColumn(tables.Column):
    def __init__(self, related_object, view=None, **kwargs):
        if view is not None and 'linkify' not in kwargs:
            url = reverse(
                'queryset_index',
                kwargs=dict(dataset=related_object.name)
            )

            def linkify(record):
                filter = {record._meta.model_name + '__pk': record.pk}
                query = view.get_query_string(ignore_original=True, **filter)
                return url + query

            kwargs.update(linkify=linkify)

        super().__init__(self, **kwargs)
        # verb name can be set after __init__, setting explicitly before
        # interferes with the automatic column class selection
        self.verbose_name = related_object.name + ' count'


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
