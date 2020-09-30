from django.db.models import DecimalField
from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.html import format_html

import django_tables2 as tables

from .models import ChangeRecord, Snapshot
from .utils import getLogger


log = getLogger('mibios')


NONE_LOOKUP = 'NULL'
ORDER_BY_FIELD = 'sort'


class CountColumn(tables.Column):
    """
    Count column

    A column showing the number of related rows from a different table with
    link to exactly those related records.  Those related records can be from a
    reverse foreign key relation or the elements of a grouped-by record.
    """
    def __init__(self, related_object=None, view=None, group_by=[],
                 force_verbose_name=True, exclude_from_export=True, **kwargs):
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
            data_name = view.data_name
            our_name = view.data_name
        else:
            # name of the relation's model
            data_name = related_object.related_model._meta.model_name
            # our_name: the name of the foreign key field of the related model
            # to the current table
            our_name = related_object.remote_field.name

        url = reverse('table', kwargs=dict(data_name=data_name))

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

                query = view.build_query_dict(filter=f).urlencode()
                if query:
                    query = '?' + query
                return url + query

            kwargs.update(linkify=linkify)

        # prepare URL for footer
        if related_object is None:
            f = view.filter.copy()
            elist = view.excludes.copy()
        else:
            via_parent = (
                issubclass(view.model, related_object.model)
                and not view.model == related_object.model
            )
            if via_parent:
                # have to add the parent between us and the relation
                # child name: how we're known to the parent
                child = related_object.model.get_child_info()[view.model].name
                our_name = our_name + '__' + child

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

        q = view.build_query_dict(filter=f, excludes=elist, negate=view.negate)
        self.footer_url = url + ('?' + q.urlencode()) if q else ''

        super().__init__(self, exclude_from_export=exclude_from_export,
                         **kwargs)

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
        """
        Sum up the counts and render a link a table with related records

        This needs to look at the whole table not just the page that is
        displayed and so needs a separate database query.
        """
        total = 0
        try:
            total = \
                table.rev_rel_counts_totals[bound_column.accessor + '__sum']
        except AttributeError as e:
            # raised if table.data.data is not a mibios.QuerySet
            log.debug('Failed getting count column totals optimized:', e)
            # try the normal way, also extra database query: retrieves the
            # whole table with annotations but the result is cached for
            # subsequent count columns
            for row in table.data:
                total += bound_column.accessor.resolve(row)

        return format_html('all: <a href={}>{}</a>', self.footer_url, total)


class ManyToManyColumn(tables.ManyToManyColumn):
    def __init__(self, *args, **kwargs):
        if 'default' not in kwargs:
            kwargs['default'] = ''

        super().__init__(*args, **kwargs)


class Table(tables.Table):
    @cached_property
    def rev_rel_counts_totals(self):
        """
        Provide the sums of count columns

        Requires the data to be backed by mibios.QuerySet.  This returns a dict
        with keys of the form <model_name>__count__sum and int values for each
        count column.
        """
        return self.data.data.sum_rev_rel_counts()


class DecimalColumn(tables.Column):
    """
    A column to display decimal numbers

    Numbers displayed in HTML will be rounded, but not numbers in exported
    data.  In HTML, cell content will be right justified.
    """
    DEFAULT_PLACES = 2
    MAX_PLACES = 2

    def __init__(self, *args, places=DEFAULT_PLACES, **kwargs):
        places = min(places, self.MAX_PLACES)  # apply cap
        self.format_string = '{:4.' + str(places) + 'f}'
        attrs = {
            # align/justify these columns to the right
            # align in <th> does not do it in Chromium, so style is used
            # not tested in other browsers
            'th': {'style': 'text-align: right;'},
            'td': {'align': 'right'},
        }
        super().__init__(*args, attrs=attrs, **kwargs)

    def render(self, value):
        return self.format_string.format(value)

    def value(self, value):
        """
        Return not rounded values for export

        Without this method, render() would be used for export
        """
        return value


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
        order_by_field=ORDER_BY_FIELD,
        fields=[],
    )
    opts = {}

    for accessor in field_names:
        try:
            field = model.get_field(accessor)
        except LookupError:
            field = None

        relations, _, name = accessor.rpartition('__')
        # django_tables2 wants dotted accessors
        col = tables.A(accessor.replace('__', '.'))
        meta_opts['fields'].append(col)
        col_kw = {}

        # preserve verbose names with odd capitalization, e.g: 'pH':
        # (By default django_tables2 uses capfirst() transforms)
        if field is not None:
            try:
                verbose_name = field.verbose_name
            except AttributeError:
                # M2M fields don't have verbose_name, but:
                verbose_name = field.related_model._meta.verbose_name

            if verbose_name[0].islower():
                if verbose_name[1].isupper():
                    col_kw['verbose_name'] = verbose_name

        if accessor == 'name':
            table_class = tables.Column
            if 'name' not in model.get_fields().names:
                # name is actually the natural property, so have to set
                # some proxy sorting, else the machinery tries to fetch the
                # 'name' column (and fails)
                if model._meta.ordering:
                    col_kw['order_by'] = model._meta.ordering
                else:
                    col_kw['order_by'] = None
            col_kw['linkify'] = True

        elif accessor == 'id':
            table_class = tables.Column
            # make one of id or name columns have an edit link / hide id if
            # name is present
            col_kw['linkify'] = 'name' not in field_names
            col_kw['visible'] = 'name' not in field_names

        # m2m fields
        elif field is not None and field.many_to_many:
            table_class = tables.ManyToManyColumn

        elif name == 'natural':
            table_class = tables.Column
            # TODO: add order by proxy
            col_kw['orderable'] = False

        # averages
        elif accessor == 'avg_group_count':
            table_class = CountColumn
            col_kw['view'] = view
            col_kw['group_by'] = view.avg_by
            col_kw['force_verbose_name'] = 'avg group count'

        elif isinstance(field, DecimalField):
            table_class = DecimalColumn
            col_kw['places'] = getattr(field, 'decimal_places',
                                       DecimalColumn.DEFAULT_PLACES)

        else:
            table_class = tables.Column

        opts[col] = table_class(**col_kw)

    if count_columns:
        # reverse relations -> count columns
        for i in model.get_related_objects():
            # col_name here must be same as what the annotation is made with
            col_name = i.related_model._meta.model_name + '__count'
            opts[col_name] = CountColumn(i, view=view)
            meta_opts['fields'].append(col_name)

    for k, v in extra.items():
        # TODO: allow specifiying the position
        meta_opts.append(k)
        opts[k] = v

    parent = Table
    Meta = type('Meta', (getattr(parent, 'Meta', object),), meta_opts)
    opts.update(Meta=Meta)

    name = 'Autogenerated' + model._meta.model_name.capitalize() + 'Table'
    return type(name, (parent, ), opts)


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
    """
    Column that lists a snapshot's tables
    """
    def __init__(self, snapshot_name, **kwargs):
        def linkify(record):
            return reverse(
                'snapshot_table',
                kwargs=dict(
                    name=snapshot_name,
                    app=record['app'],
                    table=record['table'],
                )
            )
        super().__init__(self, linkify=linkify, **kwargs)
        self.verbose_name = 'available tables'
