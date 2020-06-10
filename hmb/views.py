import csv

from django.apps import apps
from django.http import HttpResponse

from django_tables2 import SingleTableView, Table, A

from .dataset import DATASET
from .models import FecalSample


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

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        self.filter = {}
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
            self.filter.update(**dataset['filter'])
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

    def get_queryset(self):
        if self.model is None:
            return []
        else:
            return super().get_queryset().filter(**self.filter)

    def get_table_class(self):
        """
        Generate and supply table class

        overrides super
        cf. https://stackoverflow.com/questions/60311552
        """
        if self.model is None:
            return Table

        meta_opts = dict(
            model=self.model,
            template_name='django_tables2/bootstrap.html',
            fields=[A(i.replace('__', '.')) for i in self.fields],
        )
        Meta = type('Meta', (object,), meta_opts)
        name = self.dataset_name.capitalize() + 'IndexTable'
        table_opts = {'Meta': Meta}
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

        # add pk column if needed
        if 'name' not in self.fields:
            self.fields = ['id'] + self.fields
            self.col_names = \
                ['{}_id'.format(kwargs.get('dataset'))] + self.col_names

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
        r.render_row(self.col_names)
        for i in self.get_table().as_values():
            r.render_row(i)

        return response
