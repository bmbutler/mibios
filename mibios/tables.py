from django.urls import reverse
from django.utils.html import format_html

import django_tables2 as tables

from .models import ChangeRecord, Snapshot


NONE_LOOKUP = 'NULL'


class CountColumn(tables.Column):
    """
    Count column

    A column showing the number of related rows from a different table with
    link to exactly those related records.  Those related records can be from a
    reverse foreign key relation or the elements of a grouped-by record.
    """
    def __init__(self, related_object=None, view=None, group_by=[],
                 force_verbose_name=True, **kwargs):
        """
        Count column constructor

        :param related_object: The reverse relation field, i.e. an element of
                               Model._meta.related_objects
        :param list group_by: Names (str) by which the data was grouped,
                              e.g. when taking averaged
        :param force_verbose_name: Overrides the verbose column name that the
                                   django_table2 internals come up with. If
                                   this is True a verbose name will be
                                   generated and if a string is given that will
                                   be used instead.
        """
        if related_object is None:
            dataset_name = view.dataset_name
            our_name = view.dataset_name
        else:
            # name of the relation's model
            dataset_name = related_object.name
            # our_name: the name of the foreign key field of the related model
            # to the current table
            our_name = related_object.remote_field.name

        url = reverse(
            'queryset_index',
            kwargs=dict(dataset=dataset_name)
        )

        if 'linkify' not in kwargs:
            def linkify(record):
                f = {}
                for i in group_by:
                    try:
                        # ValuesIterable is used in QuerySet
                        f[i] = record[i]
                    except TypeError:
                        # records are regular instances
                        f[i] = getattr(record, i)

                if hasattr(record, 'natural'):
                    f[our_name] = record.natural

                query = view.build_query_string(filter=f)
                return url + query

            kwargs.update(linkify=linkify)

        # prepare URL for footer
        if related_object is None:
            f = view.filter.copy()
            elist = view.excludes.copy()
        else:
            f = {our_name + '__' + k: v for k, v in view.filter.items()}

            elist = []
            for i in view.excludes:
                e = {our_name + '__' + k: v for k, v in i.items()}
                if e:
                    elist.append(e)

            # if there is a filter selecting for us, then skip exclusion of
            # missing data:
            for i in f:
                if i.startswith(our_name):
                    break
            else:
                elist.append({our_name: NONE_LOOKUP})

        q = view.build_query_string(filter=f, excludes=elist,
                                    negate=view.negate)
        self.footer_url = url + q

        super().__init__(self, **kwargs)

        # django_tables2 internals somehow mess up the verbose name, but
        # verb name can be set after __init__, setting explicitly before
        # interferes with the automatic column class selection
        if force_verbose_name:
            if isinstance(force_verbose_name, str):
                self.verbose_name = force_verbose_name
            elif related_object is None:
                self.verbose_name = 'group count'
            else:
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


def table_factory(model=None, field_names=[], view=None, count_columns=True,
                  extra={}, group_by_count=None):
    """
    Generate table class from list of field/annotation/column names etc.

    :param list field_names: Names of a queryset's fields/annotations/lookups
    :param TableView view: The TableView object, will be passed to e.g.
                           CountColumn which needs various view attributes to
                           generate href urls.
    :param str group_by_count: Name of the group-by-count column in average
                                tables.  This column will get a special link to
                                the table of group members, similar to the
                                reverse relation count columns.
    """

    meta_opts = dict(
        model=model,
        template_name='django_tables2/bootstrap.html',
        fields=[],
    )
    opts = {}

    for i in field_names:
        i = tables.A(i.replace('__', '.'))
        meta_opts['fields'].append(i)

        # make one of id or name columns have an edit link
        if i == 'name':
            sort_kw = {}
            if 'name' not in model.get_fields().names:
                # name is actually the natural property, so have to set
                # some proxy sorting, else the machinery tries to fetch the
                # 'name' column (and fails)
                if model._meta.ordering:
                    sort_kw['order_by'] = model._meta.ordering
                else:
                    sort_kw['order_by'] = None
            opts[i] = tables.Column(linkify=True, **sort_kw)
        elif i == 'id':
            # hide id if name is present
            opts[i] = tables.Column(linkify='name' not in field_names,
                                    visible='name' not in field_names)

        # m2m fields
        elif i in [j.name for j in model._meta.many_to_many]:
            opts[i] = tables.ManyToManyColumn()

        elif i == 'natural':
            opts[i] = tables.Column(orderable=False)

        # averages
        elif i == 'avg_group_count':
            opts[i] = CountColumn(view=view, group_by=view.avg_by,
                                  force_verbose_name='avg group count')

    if count_columns:
        # reverse relations -> count columns
        for i in model._meta.related_objects:
            opts[i.name + '__count'] = CountColumn(i, view=view)
            meta_opts['fields'].append(i.name + '__count')

    for k, v in extra.items():
        # TODO: allow specifiying the position
        meta_opts.append(k)
        opts[k] = v

    parent = tables.Table
    Meta = type('Meta', (getattr(parent, 'Meta', object),), meta_opts)
    opts.update(Meta=Meta)

    name = 'Autogenerated' + model._meta.model_name.capitalize() + 'Table'
    klass = type(name, (parent, ), opts)
    # TODO: monkey-patching verbose_names?
    return klass


class HistoryTable(tables.Table):
    class Meta:
        model = ChangeRecord
        fields = (
            'timestamp', 'is_created', 'is_deleted', 'user', 'file.file',
            'line', 'command_line', 'fields',
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


class SnapshotListTable(tables.Table):
    """
    Table of database snapshots

    The name of each snapshotlinks to a page listing the tables available for
    that snapshot
    """
    name = tables.Column(linkify=('snapshot', {'name': tables.A('name')}))

    class Meta:
        model = Snapshot
        fields = ('timestamp', 'name', 'note')


class SnapshotTableColumn(tables.Column):
    def __init__(self, snapshot_name, **kwargs):
        def linkify(record):
            return reverse(
                'snapshot_table',
                kwargs=dict(name=snapshot_name, table=record['table'])
            )
        super().__init__(self, linkify=linkify, **kwargs)
        self.verbose_name = 'available tables'
