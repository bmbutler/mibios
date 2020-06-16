import csv
import io

from django.apps import apps
from django.db.models import Count
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.urls import reverse
from django.views.generic.edit import FormView

from django_tables2 import SingleTableView, A

from .dataset import DATASET
from .forms import UploadFileForm
from .load import GeneralLoader
from .management.import_base import AbstractImportCommand
from .models import FecalSample
from .tables import CountColumn, Table


class TestTable(Table):
    class Meta:
        model = FecalSample
        fields = ['number', 'participant', A('participant.name'),
                  A('participant.diet.dose')]


class TestView(SingleTableView):
    template_name = 'hmb/test.html'
    table_class = TestTable

    def get_queryset(self):
        return FecalSample.objects.all()


class TableView(SingleTableView):
    template_name = 'hmb/model_index.html'
    QUERY_FILTER_PREFIX = 'filter__'

    # set by setup()
    model = None
    fields = None
    col_names = None
    filter = None
    dataset_filter = None

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        self.filter = {}
        self.dataset_filter = {}
        self.fields = []
        self.col_names = []
        if 'dataset' not in kwargs:
            self.model = None
            return

        # load special dataset
        dataset = DATASET.get(kwargs['dataset'])
        if kwargs['dataset'] in DATASET:
            dataset = DATASET[kwargs['dataset']]
            self.dataset_name = kwargs['dataset']
            self.dataset_verbose_name = kwargs['dataset']
            model = dataset['model']
            self.dataset_filter = dataset['filter']
            for i in dataset['fields']:
                if isinstance(i, tuple):
                    self.fields.append(i[0])
                    self.col_names.append(i[-1])
                else:
                    # assume i is str
                    self.fields.append(i)
                    self.col_names.append(i)
        else:
            model = kwargs['dataset']

        # raises on invalid model
        self.model = apps.get_app_config('hmb').get_model(model)

        if kwargs['dataset'] not in DATASET:
            # normal model
            self.dataset_name = self.model._meta.model_name
            self.dataset_verbose_name = self.model._meta.verbose_name
            # set default fields - just the "simple" ones
            no_name_field = True
            for i in self.model.get_simple_fields():
                if i.name == 'id':
                    continue
                if i.name == 'name':
                    no_name_field = False
                self.fields.append(i.name)
                # FIXME: need this?:
                # self.col_names.append(i.verbose_name)

            if no_name_field and hasattr(self.model, 'name'):
                # add column for canonical name
                self.fields = ['name'] + self.fields

    def get(self, request, *args, **kwargs):
        self.filter.update(self.get_filter_from_url())
        return super().get(request, *args, **kwargs)

    def get_filter_from_url(self):
        """
        Compile filter dict from GET
        """
        filter = {}
        for k, v in self.request.GET.items():
            # v is last value if mutliples
            if k.startswith(self.QUERY_FILTER_PREFIX):
                # rm prefix
                k = k[len(self.QUERY_FILTER_PREFIX):]
                # last one wins
                filter[k] = v
        return filter

    def get_query_string(self, ignore_original=False, **extras):
        """
        Build the GET querystring from the user filter

        Extra filter arguments can be used to add or override the original
        filter.  With ignore_original self.filter will not be used, just the
        extras.  This is the reverse of the get_filter_from_url method
        """
        filter = {}
        if not ignore_original:
            filter.update(self.filter)
        filter.update(**extras)
        return '?' + '&'.join([
            '{}{}={}'.format(self.QUERY_FILTER_PREFIX, k, v)
            for k, v in filter.items()
        ])

    def get_queryset(self):
        if self.model is None:
            return []
        else:
            # add reverse relation count annotations
            cts = [Count(i.name) for i in self.model._meta.related_objects]
            return super() \
                .get_queryset() \
                .filter(**self.dataset_filter, **self.filter) \
                .annotate(*cts)

    def get_table_class(self):
        """
        Generate and supply table class

        overrides super
        cf. https://stackoverflow.com/questions/60311552
        """
        if self.model is None:
            return Table

        fields = [A(i.replace('__', '.')) for i in self.fields]
        exclude = []
        # conditionally exclude fields defined in the table class:
        if 'name' in self.fields:
            exclude.append('id')
        else:
            exclude.append('name')
            fields = ['id'] + fields

        # reverse relations
        table_opts = {
            i.name + '__count': CountColumn(i, view=self)
            for i in self.model._meta.related_objects
        }
        fields += [
            i.name + '__count'
            for i in self.model._meta.related_objects
        ]

        meta_opts = dict(
            model=self.model,
            template_name='django_tables2/bootstrap.html',
            fields=fields,
            exclude=exclude,
        )
        Meta = type('Meta', (object,), meta_opts)
        name = self.dataset_name.capitalize() + 'IndexTable'
        table_opts.update(Meta=Meta)
        # FIXME: call django_tables2.table_factory??
        c = type(name, (Table,), table_opts)
        # Monkey-patch column headers
        for i, j in zip(self.fields, self.col_names):
            if i != j and j:
                c.base_columns[i.replace('__', '.')].verbose_name = j
        return c

    @classmethod
    def get_model_names(cls):
        """
        Helper to get list of all model names
        """
        return sorted([
            i._meta.model_name
            for i
            in apps.get_app_config('hmb').get_models()
        ])

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.model is not None:
            ctx['model'] = self.model._meta.model_name
            ctx['dataset_name'] = self.dataset_name
            ctx['dataset_verbose_name'] = self.dataset_verbose_name
            ctx['count'] = self.get_queryset().count()
        ctx['model_names'] = self.get_model_names()
        ctx['data_sets'] = DATASET.keys()
        query = self.request.GET.urlencode()
        if query:
            ctx['query'] = '?' + query
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

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        # put model name in id column header
        if 'id' in self.fields:
            self.col_names = [
                '{}_id'.format(kwargs.get('dataset')) if i == 'id' else i
                for i in self.col_names
            ]

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
        for i in self.get_table().as_values():
            r.render_row(i)

        return response


class ImportView(FormView):
    template_name = 'hmb/import.html'
    form_class = UploadFileForm

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.dataset = kwargs.get('dataset')

    def form_valid(self, form):
        # do data import
        f = io.TextIOWrapper(form.files['file'])
        print('Importing into {}: {}'.format(self.dataset, f))
        try:
            stats = GeneralLoader.load_file(f, self.dataset,
                                            warn_on_error=True)
        except Exception as e:
            if settings.DEBUG:
                raise
            msg = ('Failed to import data in uploaded file: {}: {}'
                   ''.format(type(e).__name__, e))
            msg_level = messages.ERROR
        else:
            msg = AbstractImportCommand.format_import_stats(
                **stats,
                overwrite=True,
                verbose_changes=True,
            )
            msg_level = messages.SUCCESS

        f.close()
        messages.add_message(self.request, msg_level, msg)
        print('Import stats:\n', msg)

        return super().form_valid(form)

    def get_success_url(self):
        return reverse('queryset_index', kwargs=dict(dataset=self.dataset))
