import csv
import io

from django.apps import apps
from django.db.models import Count
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.contenttypes.models import ContentType
from django.http import Http404, HttpResponse
from django.urls import reverse
from django.utils.http import urlencode
from django.views.generic.base import ContextMixin, TemplateView
from django.views.generic.edit import FormView

from django_tables2 import SingleTableView, A, Column

from .dataset import registry
from .forms import UploadFileForm
from .load import GeneralLoader
from .management.import_base import AbstractImportCommand
from .models import Q, ChangeRecord
from .tables import (CountColumn, DeletedHistoryTable, HistoryTable,
                     ManyToManyColumn, NONE_LOOKUP, Table)
from .utils import getLogger


log = getLogger(__name__)


class UserRequiredMixin(LoginRequiredMixin):
    raise_exception = True
    permission_denied_message = 'You don\'t have an active user account here.'


class CuratorRequiredMixin(UserRequiredMixin, UserPassesTestMixin):
    group_name = 'curators'
    permission_denied_message = 'You are not a curator'
    def test_func(self):
        return self.request.user.groups.filter(name=self.group_name).exists()


class BaseMixin(ContextMixin):
    """
    Mixin to populate context for the base template
    """
    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_names'] = sorted(registry.get_model_names())
        ctx['data_sets'] = sorted(registry.get_dataset_names())
        ctx['page_title'] = apps.get_app_config('mibios').verbose_name
        return ctx


class DatasetMixin():
    """
    Mixin for views that deal with one dataset/model

    The url to which the inheriting view responds must supply a 'dataset' kwarg
    that identifies the dataset or model.
    """
    def setup(self, request, *args, **kwargs):
        """
        Set up dataset/model attributes of instance

        This overrides (but calls first) View.setup()
        """
        super().setup(request, *args, **kwargs)
        data_name = kwargs.get('dataset', None)

        self.filter = {}
        self.excludes = []
        self.dataset_filter = {}
        self.dataset_excludes = []
        self.fields = []
        self.col_names = []
        if data_name is None:
            self.model = None
            return

        # load special dataset
        try:
            dataset = registry.datasets[data_name]
        except KeyError:
            try:
                self.model = registry.models[data_name]
            except KeyError:
                raise Http404
            else:
                # setup for normal model
                self.queryset = self.model.published.all()
                self.dataset_name = self.model._meta.model_name
                self.dataset_verbose_name = self.model._meta.verbose_name
                # set default fields - just the "simple" ones
                no_name_field = True
                fields = self.model.get_fields()
                for name, verbose_name in zip(fields.names, fields.verbose):
                    if name == 'name':
                        no_name_field = False
                    self.fields.append(name)
                    if name == verbose_name:
                        # None: will be capitalized by django-tables2
                        self.col_names.append(None)
                    else:
                        # e.g. when letter case is important, like for 'pH'
                        self.col_names.append(verbose_name)
                del name, verbose_name, fields

                if no_name_field and hasattr(self.model, 'name'):
                    # add column for natural name
                    self.fields = ['name'] + self.fields
                    self.col_names = [None] + self.col_names
        else:
            # setup for special dataset
            self.dataset_name = data_name
            self.dataset_verbose_name = data_name
            self.model = dataset.model
            self.dataset_filter = dataset.filter
            self.dataset_excludes = dataset.excludes
            for i in dataset.fields:
                try:
                    fieldname, colname = i
                except ValueError:
                    # assume one-tuple
                    fieldname = i[0]
                    colname = i[0]
                except TypeError:
                    # assume i is str
                    fieldname = i
                    colname = i

                self.fields.append(fieldname)
                self.col_names.append(colname)
            del fieldname, colname

            if dataset.manager:
                self.queryset = getattr(self.model, dataset.manager).all()


class TableView(BaseMixin,DatasetMixin, UserRequiredMixin, SingleTableView):
    template_name = 'mibios/model_index.html'
    QUERY_FILTER = 'filter'
    QUERY_EXCLUDE = 'exclude'
    QUERY_NEGATE = 'inverse'
    QUERY_LIST_SEP = ','
    QUERY_KEY_VAL_SEP = ':'

    # set by setup()
    model = None
    fields = None
    col_names = None
    filter = None
    excludes = None
    negate = None
    dataset_filter = None
    dataset_excludes = None

    def get(self, request, *args, **kwargs):
        f, e, n = self.get_filter_from_url()
        self.filter.update(**f)
        self.excludes += e
        self.negate = n
        return super().get(request, *args, **kwargs)

    def get_filter_from_url(self):
        """
        Compile filter and exclude dicts from GET

        Called from get()

        Converts "NULL" to None, with exact lookup this will translate to
        SQL's "IS NULL"
        """
        filter = {}
        excludes = []
        negate = False
        for qkey, qvals  in self.request.GET.lists():
            if qkey == self.QUERY_FILTER:
                for i in qvals:
                    for j in i.split(self.QUERY_LIST_SEP):
                        k, _, v = j.partition(self.QUERY_KEY_VAL_SEP)
                        if v == NONE_LOOKUP:
                            v = None
                        try:
                            v = int(v)
                        except ValueError:
                            pass
                        filter[k] = v

            elif qkey == self.QUERY_EXCLUDE:
                for i in qvals:
                    e = {}
                    for j in i.split(self.QUERY_LIST_SEP):
                        k, _, v = j.partition(self.QUERY_KEY_VAL_SEP)
                        if v == NONE_LOOKUP:
                            v = None
                        try:
                            v = int(v)
                        except (TypeError, ValueError):
                            pass
                        e[k] = v
                    excludes.append(e)
            elif qkey == self.QUERY_NEGATE:
                negate = True

        log.debug('from GET:', filter, excludes, negate)
        return filter, excludes, negate

    def to_query_string(self, filter={}, excludes=[], negate=False,
                        without=[]):
        """
        Get query string from current state

        If negate is True, then negate the current negation state.
        Extra filters or excludes can be amended.

        :param without list: list of dicts (with kwargs of elements of
                             self.filter) and/or lists (elements of
                             self.excludes) which will be omitted from
                             the query string.
        """
        f = {**self.filter, **filter}
        elist = self.excludes + excludes

        for i in without:
            if isinstance(i, dict):
                for k, v in i.items():
                    if k in f and f[k] == v:
                        del f[k]
            elif isinstance(i, list):
                elist = [j for j in elist if i not in elist]
            else:
                raise TypeError('{} in without is neither a dict nor a list'
                                .format(i))

        if negate:
            query_negate = not self.negate
        else:
            query_negate = self.negate

        return self.build_query_string(f, elist, query_negate)

    @classmethod
    def format_query_string(cls, lookups):
        """
        Helper to format a set of lookups into query string format

        E.g. {'a': 'b', 'c': None} ==> 'a:b,c:NULL'
        """
        ret = []
        for k, v in lookups.items():
            if v is None:
                v = NONE_LOOKUP
            ret.append('{}{}{}'.format(k, cls.QUERY_KEY_VAL_SEP, v))
        return cls.QUERY_LIST_SEP.join(ret)

    @classmethod
    def build_query_string(cls, filter={}, excludes=[], negate=False):
        """
        Build the GET querystring from lookup dicts

        This is the reverse of the get_filter_from_url method
        """
        query_dict = {}
        f = cls.format_query_string(filter)
        if f:
            query_dict[cls.QUERY_FILTER] = f

        elist = []
        for i in excludes:
            e = cls.format_query_string(i)
            if e:
                elist.append(e)

        if elist:
            query_dict[cls.QUERY_EXCLUDE] = elist

        if negate:
            query_dict[cls.QUERY_NEGATE] = ''  # TODO: maybe True or Yes or so?

        query = urlencode(query_dict, doseq=True)
        if query:
            query = '?' + query
        return query

    def get_queryset(self):
        if hasattr(self, 'object_list'):
            return self.object_list

        if self.model is None:
            return []

        # add reverse relation count annotations
        cts = {}
        for i in self.model._meta.related_objects:
            kwargs = {}
            f = i.related_model.published.get_publish_filter()
            if f:
                kwargs = dict(filter=f)
            cts[i.related_model._meta.model_name + '__count'] \
                = Count(i.name, **kwargs)

        excludes = []
        for i in self.dataset_excludes + self.excludes:
            excludes.append(~Q(**i, model=self.model))

        filter = {**self.dataset_filter, **self.filter}
        q = Q(*excludes, **filter, model=self.model)

        if self.negate:
            q = ~q

        log.debug('QUERYSET FILTER:', q, 'ANNOTATION:', cts)
        return super().get_queryset().filter(q).annotate(**cts)

    def get_table_class(self):
        """
        Generate and supply table class

        overrides super
        cf. https://stackoverflow.com/questions/60311552
        """
        if self.model is None:
            return Table

        fields = [A(i.replace('__', '.')) for i in self.fields]
        table_opts = {}

        # make one of id or name columns have an edit link
        # hide id if name is present
        if 'name' in fields:
            sort_kw = {}
            if 'name' not in self.model.get_fields().names:
                # name is actually the natural property, so have to set
                # some proxy sorting, else the machinery tries to fetch the
                # 'name' column (and fails)
                if self.model._meta.ordering:
                    sort_kw['order_by'] = self.model._meta.ordering
                else:
                    sort_kw['order_by'] = None
            table_opts['name'] = Column(linkify=True, **sort_kw)
        if 'id' in fields:
            table_opts['id'] = Column(linkify='name' not in fields,
                                      visible='name' not in fields)

        # m2m fields
        for i in self.model._meta.many_to_many:
            if i.name in self.fields:
                table_opts.update({i.name: ManyToManyColumn()})

        # reverse relations
        table_opts.update({
            i.name + '__count': CountColumn(i, view=self)
            for i in self.model._meta.related_objects
        })
        fields += [
            i.name + '__count'
            for i in self.model._meta.related_objects
        ]

        if 'natural' in fields:
            table_opts['natural'] = Column(orderable=False)

        meta_opts = dict(
            model=self.model,
            template_name='django_tables2/bootstrap.html',
            fields=fields,
        )
        Meta = type('Meta', (object,), meta_opts)
        name = self.dataset_name.capitalize() + 'IndexTable'
        table_opts.update(Meta=Meta)
        # FIXME: call django_tables2.table_factory??
        c = type(name, (Table,), table_opts)
        # Monkey-patch column headers
        for i, j in zip(self.fields, self.col_names):
            if i != j and j and i != 'id':
                c.base_columns[i.replace('__', '.')].verbose_name = j
        return c

    def get_sort_by_field(self):
        field = self.request.GET.get(self.get_table()._meta.order_by_field)
        if not field:
            return None

        field = field.lstrip('-')
        if field not in self.model.get_fields().names:
            return None

        return field

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.model is None:
            return ctx

        ctx['model'] = self.model._meta.model_name
        ctx['dataset_name'] = self.dataset_name
        ctx['page_title'] += ' ' + self.dataset_name
        ctx['dataset_verbose_name'] = self.dataset_verbose_name
        ctx['count'] = self.get_queryset().count()

        ctx['applied_filter'] = [
            (k, v, self.to_query_string(without=[{k: v}]))
            for k, v
            in self.filter.items()
        ]
        ctx['applied_excludes_list'] = [
            (i, self.to_query_string(without=[i]))
            for i
            in self.excludes
        ]

        sort_by_field = self.get_sort_by_field()
        if sort_by_field is not None:
            ctx['sort_by_field'] = sort_by_field
            qs = self.get_queryset()
            stats = qs.get_field_stats(sort_by_field, natural=True)
            if 'uniform' in stats or 'unique' in stats:
                try:
                    del stats['choice_counts']
                    del stats['description']
                except KeyError:
                    pass
            else:
                # a non-boring column
                if 'description' in stats:
                    # only give these for numeric columns
                    try:
                        if stats['description'].dtype.kind == 'f':
                            del stats['choice_counts']
                        else:
                            del stats['description']
                    except KeyError:
                        pass

                filter_link_data = [
                    (
                        value,
                        count,
                        # TODO: applying filter to negated queryset is more
                        # complicated
                        self.to_query_string(filter={sort_by_field: value})
                    )
                    for value, count
                    in stats.get('choice_counts', {}).items()
                ]
                ctx['filter_link_data'] = filter_link_data
            ctx['sort_by_stats'] = stats

        query = self.request.GET.urlencode()
        if query:
            ctx['query'] = '?' + query
            ctx['invquery'] = self.to_query_string(negate=True)

        return ctx


class CSVRenderer():
    def __init__(self, response):
        self.writer = csv.writer(response, delimiter='\t')

    def render_row(self, row):
        self.writer.writerow(row)


class ExportView(TableView):
    """
    Export a queryset as file download
    """
    # Supported export format registry
    # (name, file suffix, http content type, renderer class)
    FORMATS = (
        ('csv', '.csv', 'text/csv', CSVRenderer),
    )

    def render_to_response(self, context):
        for name, suffix, content_type, renderer_class in self.FORMATS:
            if name == self.kwargs.get('format'):
                break
        else:
            raise ValueError('Export file type not supported: {}'
                             ''.format(format))

        response = HttpResponse(content_type=content_type)
        f = context['dataset_name'] + suffix
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(f)

        r = renderer_class(response)
        count_cols = [
            i.name + '__count'
            for i in self.model._meta.related_objects
        ]
        for i in self.get_table().as_values(exclude_columns=count_cols):
            r.render_row(i)

        return response


class ImportView(BaseMixin, DatasetMixin, CuratorRequiredMixin, FormView):
    template_name = 'mibios/import.html'
    form_class = UploadFileForm

    def form_valid(self, form):
        # do data import
        f = form.files['file']
        ff = io.TextIOWrapper(f)
        log.debug('Importing into {}: {}'.format(self.dataset, ff))
        try:
            stats = GeneralLoader.load_file(
                ff,
                self.dataset,
                dry_run=form.cleaned_data['dry_run'],
                can_overwrite=form.cleaned_data['overwrite'],
                warn_on_error=True,
                user=self.request.user,
            )

        except Exception as e:
            if settings.DEBUG:
                raise
            msg = ('Failed to import data in uploaded file: {}: {}'
                   ''.format(type(e).__name__, e))
            msg_level = messages.ERROR
        else:
            msg = AbstractImportCommand.format_import_stats(
                **stats,
                verbose_changes=True,
            )
            msg_level = messages.SUCCESS

        f.close()
        messages.add_message(self.request, msg_level, msg)
        log.info('IMPORT:', self.request.user, f, msg)

        return super().form_valid(form)

    def get_success_url(self):
        return reverse('queryset_index', kwargs=dict(dataset=self.dataset))

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['page_title'] += ' data import: ' + self.dataset_name
        # col_names are for django_tables2's benefit, so we need to use the
        # field names if the col name is None:
        ctx['col_names'] = [
            (j if j else i.capitalize())
            for i, j in zip(self.fields, self.col_names)
        ]
        return ctx


class HistoryView(BaseMixin, CuratorRequiredMixin, SingleTableView):
    table_class = HistoryTable

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        self.record_pk = kwargs['pk']
        try:
            # record_type: can't name this content_type, that's taken in
            # TemplateResponseMixin
            self.record_type = ContentType.objects.get_by_natural_key(
                'mibios',
                kwargs['dataset'],
            )
        except ContentType.DoesNotExist:
            raise Http404

        model_class = self.record_type.model_class()
        try:
            self.record = model_class.objects.get(pk=self.record_pk)
        except model_class.DoesNotExist:
            self.record = None

    def get_queryset(self):
        if not hasattr(self, 'object_list'):
            f = dict(
                record_type=self.record_type,
                record_pk=self.record_pk,
            )
            self.object_list = ChangeRecord.objects.filter(**f)

        return self.object_list

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['record_model'] = self.record_type.name
        if self.record:
            natural_key = self.record.natural
        else:
            try:
                natural_key = self.get_queryset().first().record_natural
            except AttributeError:
                # if no history saved, first() returns None
                natural_key = '???'
        ctx['natural_key'] = natural_key
        ctx['page_title'] += ' - history of ' + natural_key
        return ctx


class DeletedHistoryView(BaseMixin, CuratorRequiredMixin, SingleTableView):
    template_name = 'mibios/deleted_history.html'
    table_class = DeletedHistoryTable

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        try:
            # record_type: can't name this content_type, that's taken in
            # TemplateResponseMixin
            self.record_type = ContentType.objects.get_by_natural_key(
                'mibios',
                kwargs['dataset'],
            )
        except ContentType.DoesNotExist:
            raise Http404

        model_class = self.record_type.model_class()

    def get_queryset(self):
        if not hasattr(self, 'object_list'):
            f = dict(
                is_deleted=True,
                record_type=self.record_type,
            )
            self.object_list = ChangeRecord.objects.filter(**f)

        return self.object_list

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['record_model'] = self.record_type.name
        ctx['page_title'] += ' - deleted records'
        return ctx


class FrontPageView(BaseMixin, UserRequiredMixin, TemplateView):
    template_name = 'mibios/frontpage.html'
    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['counts'] = {}
        models = registry.get_models()
        for i in sorted(models, key=lambda x: x._meta.verbose_name):
            count = i.objects.count()
            if count:
                ctx['counts'][i._meta.verbose_name] = count

        try:
            ctx['latest'] = ChangeRecord.objects.latest()
        except ChangeRecord.DoesNotExist:
            ctx['latest'] = None
        return ctx
