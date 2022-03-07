from collections import OrderedDict
from itertools import chain

from django_tables2 import SingleTableView
import pandas

from django.conf import settings
from django.core.management import call_command
from django.db.models import URLField
from django.urls import reverse
from django.views.generic import DetailView

from mibios_omics.models import Sample, TaxonAbundance
from . import models
from .tables import DatasetTable, SampleTable


class GlamrDetailView(DetailView):
    template_name = 'mibios_glamr/detail.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        afields = self.model.get_accession_fields()
        ctx['object_name'] = '|'.join(
            [str(getattr(self.object, i.name, None)) for i in afields]
        )
        ctx['model_name'] = self.model._meta.verbose_name
        ctx['details'] = self.get_details().values()
        return ctx

    def get_details(self):
        details = OrderedDict()
        for i in self.model._meta.get_fields():
            if i.many_to_many or i.one_to_many or i.name == 'id':
                continue
            value = getattr(self.object, i.name, None)
            if value:
                if i.many_to_one or i.one_to_one:  # TODO: test 1-1 fields
                    url = value.get_absolute_url()
                elif isinstance(i, URLField):
                    url = value
                else:
                    url = None
            else:
                url = None
            # some relations (e.g.: 1-1) don't have a verbose name:
            name = getattr(i, 'verbose_name', i.name)
            details[i.name] = (name, url, value)
        return details


class DatasetView(GlamrDetailView):
    model = models.Dataset

    def get_object(self):
        if self.kwargs.get(self.pk_url_kwarg) == 0:
            return models.Dataset.orphans
        return super().get_object()

    def get_details(self):
        details = super().get_details()
        if self.object.orphan_group:
            pk = 0
        else:
            pk = self.object.pk
        details['samples'] = (
            'Samples',
            reverse('dataset_sample_list', args=[pk]),
            f'List of {self.object.samples().count()} samples'
        )
        details.move_to_end('samples', last=False)
        return details


class DemoFrontPageView(SingleTableView):
    model = models.Dataset
    template_name = 'mibios_glamr/demo_frontpage.html'
    table_class = DatasetTable

    def get_table_data(self):
        data = super().get_table_data()
        return chain([models.Dataset.orphans], data)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['mc_abund'] = TaxonAbundance.objects \
            .filter(taxname='MICROCYSTIS') \
            .select_related('sample')[:5]
        self.make_ratios_plot()
        return ctx

    def make_ratios_plot(self):
        imgpath = settings.MEDIA_ROOT + 'var/mappedratios.png'
        ratios = pandas.DataFrame([
            (i.reads_mapped_contigs / i.read_count,
             i.reads_mapped_genes / i.read_count)
            for i in Sample.objects.all()
        ], columns=['contigs', 'genes'])
        plot = ratios.plot(x='contigs', y='genes', kind='scatter')
        plot.figure.savefig(imgpath)


class ModelGraphView(TemplateView):
    template_name = 'mibios_glamr/model_graphs.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['graphs'] = self.make_graphs()
        return ctx

    def make_graphs(self):
        apps = ['mibios_umrad', 'mibios_omics', 'mibios_glamr']
        graphs = {i: f'var/{i}.png' for i in apps}
        if 'django_extensions' in settings.INSTALLED_APPS:
            for app_name, output in graphs.items():
                call_command(
                    'graph_models',
                    app_name,
                    output=settings.MEDIA_ROOT + output,
                    exclude_models=['Model'],
                    no_inheritance=True,
                )
        return graphs


class ReferenceView(GlamrDetailView):
    model = models.Reference


class SampleListView(SingleTableView):
    """ List of samples belonging to a given dataset  """
    model = Sample
    template_name = 'mibios_glamr/sample_list.html'
    table_class = SampleTable

    def get_queryset(self):
        pk = self.kwargs['pk']
        if pk == 0:
            self.dataset = models.Dataset.orphans
        else:
            self.dataset = models.Dataset.objects.get(pk=pk)
        return self.dataset.samples()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['dataset'] = str(self.dataset)
        return ctx


class SampleView(GlamrDetailView):
    model = Sample
