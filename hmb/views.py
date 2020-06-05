import csv

from django.apps import apps
from django.http import HttpResponse

from django_tables2 import SingleTableView, Table


class TableView(SingleTableView):
    template_name = 'hmb/model_index.html'
    QUERY_FILTER_PREFIX = 'filter__'

    # set by setup()
    model = None
    filter = dict()

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.compile_filter()
        model = kwargs.get('model')
        if model:
            # raises on invalid model
            self.model = apps.get_app_config('hmb').get_model(model)

    def compile_filter(self):
        """
        Compile filter dict from GET
        """
        self.filter = {}
        for k, v in self.request.GET.items():
            # v is last value if mutliples
            if k.startswith(self.QUERY_FILTER_PREFIX):
                # rm prefix
                k = k[len(self.QUERY_FILTER_PREFIX):]
                # last one wins
                self.filter[k] = v

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
            fields=[i.name for i in self.model._meta.fields],
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
            ctx['model_name'] = self.model._meta.verbose_name
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
        qs = super().get_queryset()
        sort = self.request.GET.get('sort')
        if sort:
            qs = qs.order_by(sort)
        return qs

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
        head = None
        for i in self.get_queryset():
            row = i.export_dict()
            if head is None:
                head = list(row.keys())
                r.render_row(head)
            r.render_row(row.values())

        return response
