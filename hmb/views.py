import csv

from django.apps import apps
from django.http import HttpResponse

from django_tables2 import SingleTableView, Table

from .models import FecalSample

DATASET = {}


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
    filter = None

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)

        self.filter = {}
        self.fields = []
        if 'dataset' not in kwargs:
            self.model = None
            return

        # load special dataset
        dataset = DATASET.get(kwargs['dataset'])
        if kwargs['dataset'] in DATASET:
            dataset = DATASET[kwargs['dataset']]
            model = dataset['model']
            self.filter.update(**dataset['filter'])
            self.fields.append(dataset['fields'])
        else:
            model = kwargs['dataset']

        # raises on invalid model
        self.model = apps.get_app_config('hmb').get_model(model)

        # set default fields - just the "simple" ones
        for i in self.model.get_simple_fields():
            if i.name == 'id':
                continue
            if i.name not in self.fields:
                self.fields.append(i.name)

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

        # FIXME: there is also _meta.get_fields() ?
        meta_opts = dict(
            model=self.model,
            template_name='django_tables2/bootstrap.html',
            fields=self.fields,
        )
        Meta = type('Meta', (object,), meta_opts)
        name = self.model._meta.label.capitalize() + 'IndexTable'
        table_opts = {'Meta': Meta}
        return type(name, (Table,), table_opts)

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
            ctx['model_verbose'] = self.model._meta.verbose_name
            ctx['count'] = self.get_queryset().count()
        ctx['model_names'] = self.get_model_names()
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

    def get_queryset(self):
        """
        Add order to queryset

        Re-implements what Table does
        """
        qs = super().get_queryset()
        sort = self.request.GET.get('sort')
        if sort:
            qs = qs.order_by(sort)
        return qs.values_list(*self.fields)

    def render_to_response(self, context):
        for name, suffix, content_type, renderer_class in self.FORMATS:
            if name == self.kwargs.get('format'):
                break
        else:
            raise ValueError('Export file type not supported: {}'
                             ''.format(format))

        response = HttpResponse(content_type=content_type)
        f = context['model_name'] + suffix
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(f)

        r = renderer_class(response)
        qs = self.get_queryset()
        r.render_row(qs._fields)  # header row
        for i in self.get_queryset():
            r.render_row(i)

        return response
