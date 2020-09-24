import csv
from io import StringIO
from math import isnan
from zipfile import ZipFile, ZIP_DEFLATED

from django.apps import apps
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.contenttypes.models import ContentType
from django.http import Http404, HttpResponse
from django.http.request import QueryDict
from django.urls import reverse
from django.utils.text import slugify
from django.views.generic.base import ContextMixin, TemplateView, View
from django.views.generic.edit import FormMixin, FormView

from django_tables2 import SingleTableView, Column

from . import (__version__, QUERY_FILTER, QUERY_EXCLUDE, QUERY_NEGATE,
               QUERY_FIELD, QUERY_FORMAT, QUERY_EXPAND, get_registry)
from .forms import get_export_form, get_field_search_form, UploadFileForm
from .load import Loader
from .management.import_base import AbstractImportCommand
from .models import Q, ChangeRecord, ImportFile, Snapshot
from .tables import (DeletedHistoryTable, HistoryTable, NONE_LOOKUP,
                     SnapshotListTable, SnapshotTableColumn, Table,
                     table_factory)
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


class BasicBaseMixin(ContextMixin):
    """
    Mixin to populate context for the base template without model/dataset info
    """
    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        # page_title: a list, inheriting views should consider adding to this
        ctx['page_title'] = [getattr(
                get_registry(),
                'verbose_name',
                apps.get_app_config('mibios').verbose_name
        )]
        ctx['user_is_curator'] = \
            self.request.user.groups.filter(name='curators').exists()
        ctx['version_info'] = {'mibios': __version__}
        for conf in get_registry().apps.values():
            ctx['version_info'][conf.name] = getattr(conf, 'version', None)
        return ctx


class BaseMixin(BasicBaseMixin):
    """
    Mixin to populate context for the base template
    """
    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_names'] = []
        for conf in get_registry().apps.values():
            names = sorted(get_registry().get_model_names(app=conf.name))
            if names:
                ctx['model_names'].append((conf.verbose_name, names))
        ctx['data_names'] = []
        for conf in get_registry().apps.values():
            names = sorted(get_registry().get_dataset_names(app=conf.name))
            if names:
                ctx['data_names'].append((conf.verbose_name, names))
        ctx['snapshots_exist'] = Snapshot.objects.exists()
        return ctx


class DatasetMixin():
    """
    Mixin for views that deal with one dataset/model

    The url to which the inheriting view responds must supply a 'dataset' kwarg
    that identifies the dataset or model.
    """
    data_name = None

    def setup(self, request, *args, data_name=None, **kwargs):
        """
        Set up dataset/model attributes of instance

        This overrides (but calls first) View.setup()
        """
        super().setup(request, *args, **kwargs)
        if data_name:
            self.data_name = data_name

        self.filter = {}
        self.excludes = []
        self.dataset_filter = {}
        self.dataset_excludes = []
        self.fields = []
        self.col_names = []

        if self.data_name is None:
            # no models/datasets defined
            self.model = None
            return

        # load special dataset
        try:
            dataset = get_registry().datasets[self.data_name]
        except KeyError:
            try:
                self.model = get_registry().models[self.data_name]
            except KeyError:
                raise Http404
            else:
                # setup for normal model
                self.queryset = self.model.published.all()
                self.data_name_verbose = self.model._meta.verbose_name
                # set default fields - just the "simple" ones
                no_name_field = True
                fields = self.model.get_fields(with_m2m=True)
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
            self.data_name_verbose = self.data_name
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


class TableViewPlugin():
    """
    Parent class for per-model view plugins

    The table view plugin mechanism allows to add per-model content to the
    regular table display.  An implementation should declare a class in a
    registered app's view module that inherits from TableViewPlugin and that
    should set the model_class and template_name variables and override
    get_context_data() as needed.  The template with the to-be-added content
    needs to be provided at the app's usual template location.  The plugin's
    content will be rendered just above the table.
    """
    model_class = None
    template_name = None

    def get_context_data(self, **ctx):
        return ctx


class TableView(BaseMixin, DatasetMixin, UserRequiredMixin, SingleTableView):
    template_name = 'mibios/table.html'

    # set by setup()
    model = None
    avg_by = None
    fields = None
    col_names = None
    filter = None
    excludes = None
    negate = None
    dataset_filter = None
    dataset_excludes = None

    def get(self, request, *args, **kwargs):
        log.debug('GET query string:', request.GET)
        self.update_state(*self.compile_state_params())
        return super().get(request, *args, **kwargs)

    def update_state(self, filter, excludes, negate, fields_selected, expand):
        """
        Update state from info compiled from GET query string
        """
        self.filter.update(**filter)
        self.excludes += excludes
        self.negate = negate

        for accessor in sorted(expand, key=lambda x: len(x)):
            # order: process short field names first,i.e. deeper relations last
            try:
                rel_model = self.model.get_field(accessor).related_model
            except Exception as e:
                # accessor does not point to a field
                raise Http404(e) from e

            if rel_model is None:
                raise Http404('is not a relation: {}'.format(accessor))

            rel_fields = rel_model.get_fields(skip_auto=True).names
            rel_fields = [
                accessor + '__' + i
                for i in rel_fields
            ]

            try:
                # place expansion right of relation
                idx = self.fields.index(accessor) + 1
            except ValueError:
                # not in field selection, append to end
                idx = len(self.fields)

            self.fields[idx:idx] = rel_fields
            self.col_names[idx:idx] = rel_fields

        if fields_selected:
            # column/field selection was supplied, have to update
            # but keep order, keep sync with verbose column names
            fields, col_names = [], []
            for i, j in zip(self.fields, self.col_names):
                if i in fields_selected:
                    fields.append(i)
                    col_names.append(j)
            self.fields = fields
            self.col_names = col_names

    def compile_state_params(self):
        """
        Compile filter and field selection state, etc from GET querystring

        Called from get()

        Converts "NULL" to None, with exact lookup this will translate to
        SQL's "IS NULL"
        """
        filter = {}
        excludes = {}
        negate = False
        fields = []
        expand = []
        for qkey, val_list in self.request.GET.lists():
            if qkey.startswith(QUERY_FILTER + '-'):
                _, _, filter_key = qkey.partition('-')
                val = val_list[-1]
                if val == NONE_LOOKUP:
                    val = None
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                filter[filter_key] = val

            elif qkey.startswith(QUERY_EXCLUDE + '-'):
                _, idx, exclude_key = qkey.split('-')
                val = val_list[-1]
                if val == NONE_LOOKUP:
                    val = None
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                if idx not in excludes:
                    excludes[idx] = {}
                excludes[idx][exclude_key] = val

            elif qkey == QUERY_NEGATE:
                val = val_list[-1]
                if val:
                    negate = True
            elif qkey == QUERY_FIELD:
                fields += val_list
            elif qkey == QUERY_EXPAND:
                expand += val_list
            else:
                # unrelated item
                pass

        # convert excludes into list, forget the index
        excludes = [i for i in excludes.values()]
        log.debug('DECODED FROM QUERYSTRING:', filter, excludes, negate,
                  fields, expand)
        return filter, excludes, negate, fields, expand

    def to_query_dict(self, filter={}, excludes=[], negate=False, without=[],
                      fields=[], keep_other=False):
        """
        Compile a query dict from current state

        If negate is True, then negate the current negation state.
        Extra filters or excludes can be amended.

        :param without list: list of dicts (with kwargs of elements of
                             self.filter) and/or lists (elements of
                             self.excludes) which will be omitted from
                             the query string.
        :param list fields: List of fields to request. If an empty list, the
                            default, is passed, then nothing will be added to
                            the query string, with the intended meaning to then
                            show all the fields.
        :param bool keep_other: Return additional items, present in the
                                original request query dict but unrelated to
                                TableView, e.g. those items handled by
                                django_tables2 (sort, pagination.)
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

        if f or elist:
            if negate:
                query_negate = not self.negate
            else:
                query_negate = self.negate
        else:
            # no filtering is in effect, thus result inversion makes no sense
            query_negate = False

        qdict = self.build_query_dict(f, elist, query_negate, fields)

        if keep_other:
            for k, vs in self.request.GET.lists():
                if k not in qdict:
                    qdict.setlist(k, vs)

        return qdict

    def to_query_string(self, *args, **kwargs):
        """
        Build the GET querystring from current state

        Accepts the same arguments as to_query_dict()
        """
        query = self.to_query_dict(*args, **kwargs).urlencode()
        if query:
            query = '?' + query
        return query

    @classmethod
    def build_query_dict(cls, filter={}, excludes=[], negate=False, fields=[]):
        """
        Build a query dict

        This is the reverse of the get_filter_from_url method.  This is a class
        method so we can build arbitrary query strings.  Use
        TableView.to_query_dict() to build a query string corresponding to
        the current view.
        """
        query_dict = QueryDict(mutable=True)
        for k, v in filter.items():
            k = slugify((QUERY_FILTER, k))
            if v is None:
                v = NONE_LOOKUP
            query_dict[k] = v

        for i, excl in enumerate(excludes):
            for k, v in excl.items():
                k = slugify((QUERY_EXCLUDE, i, k))
                if v is None:
                    v = NONE_LOOKUP
                query_dict[k] = v

        if negate:
            query_dict[QUERY_NEGATE] = negate

        if fields:
            query_dict.setlist(QUERY_FIELD, fields)

        return query_dict

    def get_queryset(self):
        if hasattr(self, 'object_list'):
            return self.object_list

        if self.model is None:
            return []

        excludes = []
        for i in self.dataset_excludes + self.excludes:
            excludes.append(~Q(**i, model=self.model))

        filter = {**self.dataset_filter, **self.filter}
        q = Q(*excludes, **filter, model=self.model)

        if self.negate:
            q = ~q

        log.debug('QUERYSET FILTER:', q)
        qs = super().get_queryset().filter(q)
        # Do not annotate with rev rel counts on the average table.  Doing so
        # will mess up the group count in some circumstances (group members
        # each counted multiply times (for each rev rel count))
        if getattr(self, 'avg_by', None):
            qs = qs.average(*self.avg_by)
        else:
            qs = qs.annotate_rev_rel_counts()
        return qs

    def get_table_class(self):
        t = table_factory(model=self.model, field_names=self.fields, view=self,
                          count_columns=True)
        return t

    def get_sort_by_field(self):
        """
        Returns name of valid sort-by fields from the querystring

        If the sort-by field is not a field in the current table view None is
        returned.
        """
        field = self.request.GET.get(self.get_table()._meta.order_by_field)
        if not field:
            return None

        field = field.lstrip('-')
        # reverse from django_tables2 accessor sep
        field = field.replace('.', '__')
        if field in self.fields:
            return field

        return None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.model is None:
            return ctx

        try:
            plugin_class = \
                get_registry().table_view_plugins[self.model._meta.model_name]
        except KeyError:
            ctx['table_view_plugin_template'] = None
        else:
            plugin = plugin_class()
            ctx['table_view_plugin_template'] = plugin.template_name
            ctx = plugin.get_context_data(**ctx)

        ctx['model'] = self.model._meta.model_name
        ctx['data_name'] = self.data_name
        ctx['page_title'].append(self.data_name)
        ctx['data_name_verbose'] = self.data_name_verbose
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
        if sort_by_field is None:
            ctx['sort_by_stats'] = {}
        else:
            ctx['sort_by_field'] = sort_by_field
            qs = self.get_queryset()
            stats = qs.get_field_stats(sort_by_field, natural=True)
            if 'uniform' in stats or 'unique' in stats:
                try:
                    del stats['choice_counts']
                    del stats['description']
                except KeyError:
                    pass

                if 'unique' in stats:
                    ctx['field_search_form'] = \
                        get_field_search_form(sort_by_field)()
            else:
                # a non-boring column
                if 'description' in stats:
                    # only give these for numeric columns
                    try:
                        if stats['description'].dtype.kind == 'f':
                            # keep description and only give NaNs as filter
                            # choice
                            try:
                                nan_ct = stats['choice_counts'][[float('nan')]]
                            except KeyError:
                                del stats['choice_counts']
                            else:
                                stats['choice_counts'] = nan_ct
                        else:
                            del stats['description']
                    except KeyError:
                        pass

                filter_link_data = []
                if 'choice_counts' in stats:
                    counts = {
                        None if isinstance(k, float) and isnan(k) else k: v
                        for k, v in
                        stats['choice_counts'].items()
                    }
                    filter_link_data = [
                        (
                            value,
                            count,
                            # TODO: applying filter to negated queryset is more
                            # complicated
                            self.to_query_string(filter={sort_by_field: value})
                        )
                        for value, count
                        in counts.items()
                    ]
                ctx['filter_link_data'] = filter_link_data
            ctx['sort_by_stats'] = stats

        # the original querystring:
        query = self.request.GET.urlencode()
        if query:
            ctx['querystr'] = '?' + query
            ctx['invquery'] = self.to_query_string(negate=True)
        else:
            ctx['querystr'] = ''

        ctx['avg_by_data'] = {'-'.join(i): i for i in self.model.average_by}

        return ctx


class CSVRenderer():
    content_type = 'text/csv'
    delimiter = ','

    def __init__(self, response, **kwargs):
        self.response = response

    def render(self, values):
        """
        Render all rows to the response
        """
        writer = csv.writer(self.response, delimiter=self.delimiter)
        for i in values:
            writer.writerow(i)


class CSVTabRenderer(CSVRenderer):
    delimiter = '\t'


class CSVRendererZipped():
    content_type = 'application/zip'

    def __init__(self, response, filename):
        self.response = response
        self.filename = filename[:-len('.zip')]

    def render(self, values):
        """
        Render all rows to the response
        """
        buf = StringIO()
        writer = csv.writer(buf, delimiter='\t')
        for row in values:
            writer.writerow(row)

        buf.seek(0)
        with ZipFile(self.response, 'w', ZIP_DEFLATED) as f:
            f.writestr(self.filename, buf.read())


class ExportBaseMixin:
    # Supported export format registry
    # (name, file suffix, renderer class)
    FORMATS = (
        ('csv', '.csv', CSVRenderer),
        ('tab', '.csv', CSVTabRenderer),
        ('csv/zipped', '.csv.zip', CSVRendererZipped),
    )
    DEFAULT_FORMAT = 'csv'

    def get_format(self):
        """
        Get export file format
        """
        fmt_name = self.request.GET.get(QUERY_FORMAT)
        for i in self.FORMATS:
            if fmt_name == i[0]:
                return i

        if fmt_name:
            raise Http404('unknown export format')

        for i in self.FORMATS:
            if self.DEFAULT_FORMAT == i[0]:
                return i
        else:
            raise RuntimeError('no valid default export format defined')


class ExportMixin(ExportBaseMixin):
    """
    Export table data as file download

    Requires kwargs['format'] to be set by url conf.

    Implementing views need to provide a get_values() method that provides the
    data to be exported as an iterable over rows (which are lists of values).
    The first row should contain the column headers.
    """

    def get_filename(self):
        """
        Get the user-visible name (stem) for the file downloaded.

        The default implementation generates a default value from the registry
        name.  The returnd value is without suffix.  The suffix is determined
        by the file format.
        """
        return get_registry().name + '_data'

    def render_to_response(self, context):
        name, suffix, renderer_class = self.get_format()

        response = HttpResponse(content_type=renderer_class.content_type)
        filename = self.get_filename() + suffix
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        renderer_class(response, filename=filename).render(self.get_values())

        return response


class ExportView(ExportMixin, TableView):
    """
    File download of table data
    """
    def get_filename(self):
        return self.data_name

    def get_values(self):
        # do not export count columns
        count_cols = [
            i.name + '__count'
            for i in self.model.get_related_objects()
        ]
        return self.get_table().as_values(exclude_columns=count_cols)


class ExportFormView(ExportBaseMixin, FormMixin, TableView):
    """
    Provide the export format selection form

    The form will be submitted via GET and the query string language used, is
    what TableView expects.  So this will not use Django's usual form
    processing.
    """
    template_name = 'mibios/export.html'
    export_url_name = 'export'

    def get_form_class(self):
        initial = []
        for i in self.model.get_fields(skip_auto=True).names:
            # FIXME: this gets the wrong fields with AverageMixin since the
            # fields change during the call to AverageMixin.get_table_class()
            if i in self.fields:
                initial.append(i)
        if 'id' in self.fields and 'name' not in initial:
            # prefer name over id
            initial.append('id')

        return get_export_form(
            self.to_query_dict(fields=self.fields, keep_other=True),
            self.FORMATS,
            initial,
        )

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['page_title'].append('export')
        return ctx


class ImportView(BaseMixin, DatasetMixin, CuratorRequiredMixin, FormView):
    template_name = 'mibios/import.html'
    form_class = UploadFileForm
    log = getLogger('dataimport')

    def form_valid(self, form):
        # do data import
        f = form.files['file']
        dry_run = form.cleaned_data['dry_run']
        if dry_run:
            log.debug(
                '[dry run] Importing into {}: {}'.format(self.data_name, f)
            )
        else:
            self.log.info(
                'Importing into {}: {}'.format(self.data_name, f)
            )

        try:
            stats = Loader.load_file(
                f,
                self.data_name,
                dry_run=dry_run,
                can_overwrite=form.cleaned_data['overwrite'],
                erase_on_blank=form.cleaned_data['erase_on_blank'],
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
        finally:
            f.close()

        messages.add_message(self.request, msg_level, msg)
        args = (msg_level, 'user:', self.request.user, 'file:', f, '\n', msg)
        if dry_run:
            log.log(*args)
        else:
            self.log.log(*args)

        return super().form_valid(form)

    def get_success_url(self):
        return reverse('table', kwargs=dict(data_name=self.data_name))

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['page_title'].append('file upload')
        # col_names are for django_tables2's benefit, so we need to use the
        # field names if the col name is None:
        ctx['col_names'] = [
            (j if j else i.capitalize())
            for i, j in zip(self.fields, self.col_names)
        ]
        return ctx


class HistoryView(BaseMixin, CuratorRequiredMixin, SingleTableView):
    table_class = HistoryTable
    record = None

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        if 'record' in kwargs:
            # coming in through mibios.ModelAdmin history
            self.record = kwargs['record']
            self.record_pk = self.record.pk
            self.record_natural = self.record.natural
            app_label = self.record._meta.app_label
            data_name = self.record._meta.model_name
        else:
            # via other url conf, NOTE: has no current users
            try:
                self.record_pk = int(kwargs['natural'])
                self.record_natural = None
            except ValueError:
                self.record_pk = None
                self.record_natural = kwargs['natural']

            data_name = kwargs['dataset']
            try:
                model_class = get_registry().models[data_name]
            except KeyError:
                raise Http404
            else:
                app_label = model_class._meta.app_label

        try:
            # record_type: can't name this content_type, that's taken in
            # TemplateResponseMixin
            self.record_type = ContentType.objects.get_by_natural_key(
                app_label,
                data_name,
            )
        except ContentType.DoesNotExist:
            raise Http404

        if self.record is None:
            model_class = self.record_type.model_class()
            get_kw = {}
            if self.record_natural:
                get_kw['natural'] = self.record_natural
            elif self.record_pk:
                get_kw['pk'] = self.record_pk

            try:
                self.record = model_class.objects.get(**get_kw)
            except (model_class.DoesNotExist,
                    model_class.MultipleObjectsReturned):
                self.record = None

        if kwargs.get('extra_context'):
            if self.extra_context is None:
                self.extra_context = kwargs['extra_context']
            else:
                self.extra_context.update(kwargs['extra_context'])

    def get_queryset(self):
        if not hasattr(self, 'object_list'):
            f = dict(
                record_type=self.record_type,
            )
            if self.record_natural:
                f['record_natural'] = self.record_natural
            elif self.record_pk:
                f['record_pk'] = self.record_pk

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
        ctx['page_title'].append('history of ' + natural_key)
        return ctx


class DeletedHistoryView(BaseMixin, CuratorRequiredMixin, SingleTableView):
    template_name = 'mibios/deleted_history.html'
    table_class = DeletedHistoryTable

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        try:
            model = get_registry().models[kwargs['dataset']]
        except KeyError:
            raise Http404

        try:
            # record_type: can't name this content_type, that's taken in
            # TemplateResponseMixin
            self.record_type = ContentType.objects.get_by_natural_key(
                model._meta.app_label,
                model._meta.model_name,
            )
        except ContentType.DoesNotExist:
            raise Http404

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
        ctx['page_title'].append('deleted records')
        return ctx


class FrontPageView(BaseMixin, UserRequiredMixin, TemplateView):
    template_name = 'mibios/frontpage.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['counts'] = {}
        models = get_registry().get_models()
        for i in sorted(models, key=lambda x: x._meta.verbose_name):
            count = i.objects.count()
            if count:
                ctx['counts'][i._meta.verbose_name] = count

        try:
            ctx['latest'] = ChangeRecord.objects.latest()
        except ChangeRecord.DoesNotExist:
            ctx['latest'] = None

        ctx['admins'] = settings.ADMINS
        return ctx


class SnapshotListView(BasicBaseMixin, UserRequiredMixin, SingleTableView):
    """
    View presenting a list of snapshots with links to SnapshotView
    """
    model = Snapshot
    table_class = SnapshotListTable


class SnapshotView(BasicBaseMixin, UserRequiredMixin, SingleTableView):
    """
    View of a single snapshot, displays the list of available tables
    """
    template_name = 'mibios/snapshot.html'

    def get_table_class(self):
        meta_opts = dict(
            # model=self.model,
            # template_name='django_tables2/bootstrap.html',
        )
        Meta = type('Meta', (object,), meta_opts)
        table_opts = dict(Meta=Meta)
        table_opts.update(table=SnapshotTableColumn(self.snapshot.name))
        name = ''.join(self.snapshot.name.split()).capitalize()
        name += 'SnapshotTable'
        # FIXME: call django_tables2.table_factory??
        klass = type(name, (Table,), table_opts)
        return klass

    def get(self, request, *args, **kwargs):
        try:
            self.snapshot = Snapshot.objects.get(name=kwargs['name'])
        except Snapshot.DoesNotExist:
            raise Http404

        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        return self.snapshot.get_table_name_data()


class SnapshotTableView(BasicBaseMixin, UserRequiredMixin, SingleTableView):
    """
    Display one table from a snapshot (with all data)
    """
    template_name = 'mibios/snapshot_table.html'

    def get(self, request, *args, **kwargs):
        snapshot = kwargs['name']
        self.app_label = kwargs['app']
        self.table_name = kwargs['table']
        try:
            self.snapshot = Snapshot.objects.get(name=snapshot)
        except Snapshot.DoesNotExist:
            raise Http404

        try:
            self.columns, rows = \
                self.snapshot.get_table_data(self.app_label, self.table_name)
        except ValueError:
            # invalid table name
            raise Http404

        self.queryset = [dict(zip(self.columns, i)) for i in rows]

        return super().get(request, *args, **kwargs)

    def get_table_class(self):
        meta_opts = dict()
        Meta = type('Meta', (object,), meta_opts)
        table_opts = dict(Meta=Meta)
        for i in self.columns:
            table_opts.update(**{i: Column()})
        name = ''.join(self.snapshot.name.split()).capitalize()
        name += 'SnapshotTableTable'
        # FIXME: call django_tables2.table_factory??
        klass = type(name, (Table,), table_opts)
        return klass


class ExportSnapshotTableView(ExportMixin, SnapshotTableView):
    def get_filename(self):
        return self.snapshot.name + '_' + self.table_name

    def get_values(self):
        return self.get_table().as_values()


class ImportFileDownloadView(CuratorRequiredMixin, View):
    """
    Reply to file download request with X-Sendfile headed response
    """
    def get(self, request, *args, **kwargs):
        path = 'imported/' + str(kwargs['year']) + '/' + kwargs['name']
        try:
            file = ImportFile.objects.get(file=path)
        except ImportFile.DoesNotExist:
            raise Http404
        res = HttpResponse(content_type='')
        res['X-Sendfile'] = str(file)
        return res


class AverageMixin():
    """
    Add to TableView to display tables with averages
    """
    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        avg_by = kwargs['avg_by'].split('-')
        if avg_by == ['']:
            # for testing?
            self.avg_by = []
        else:
            for i in self.model.average_by:
                if set(avg_by) == set(i):
                    self.avg_by = avg_by
                    break
            else:
                raise Http404

    def get_table_class(self):
        """
        Generate django_tables2 table class
        """
        self.fields = self.get_queryset()._avg_fields
        t = table_factory(model=self.model, field_names=self.fields, view=self,
                          count_columns=False)
        return t

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['avg_url_slug'] = '-'.join(self.avg_by)
        ctx['page_title'].append('average')
        return ctx


class AverageView(AverageMixin, TableView):
    pass


class AverageExportView(AverageMixin, ExportView):
    pass


class AverageExportFormView(AverageMixin, ExportFormView):
    export_url_name = 'average_export'

    def get_form_class(self):
        # Have to get correct fields for this
        self.fields = self.get_queryset()._avg_fields
        return super().get_form_class()
