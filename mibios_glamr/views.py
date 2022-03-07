from collections import OrderedDict
from itertools import chain

from django_tables2 import SingleTableView
import pandas

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.core.management import call_command
from django.db.models import URLField
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse
from django.views.generic import DetailView
from django.views.generic.base import TemplateView

from mibios_omics.models import Sample, TaxonAbundance
from mibios_umrad.models import (
    CompoundEntry, CompoundName, FunctionName, Location, Metal, FuncRefDBEntry,
    ReactionEntry, TaxName, Taxon, Uniprot, UniRef100,
)
from . import models
from .forms import SearchForm
from .tables import DatasetTable, SampleTable


class AbundanceView(SingleTableView):
    # FIXME: should be a SingleObject??? something
    template_name = 'mibios_glamr/abundance.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        model_name = self.kwargs['model_name']
        for i in SearchResultView.searchables:
            if i._meta.model_name == model_name:
                self.object_model = i
                break
        else:
            raise Http404(f'no such model: {model_name}')

        accession = self.request.GET['id'].strip()

        if self.object_model == TaxName:
            acc = 'name'
        else:
            acc = self.object_model.get_accession_lookup_single()
        kw = {acc: accession}
        self.object = self.object_model.objects.get(**kw)
        self.model = self.object.abundance.model

    def get_queryset(self):
        return self.object.abundance.all()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_name_verbose'] = self.model._meta.verbose_name
        ctx['object'] = self.object

        return ctx


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
        ctx['search_form'] = SearchForm()
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


class SearchResultView(TemplateView):
    template_name = 'mibios_glamr/search_results.html'

    searchables = [
        TaxName, Taxon, CompoundEntry, ReactionEntry, UniRef100, CompoundName,
        FunctionName, Location, Metal, FuncRefDBEntry, Uniprot, Sample,
    ]

    def get(self, request, *args, **kwargs):
        self.search()
        if self.results:
            ctx = self.get_context_data(
                search_results=self.results,
                query=self.query,
            )
        else:
            return HttpResponseRedirect(reverse('frontpage'))
        return self.render_to_response(ctx)

    def search(self, abundance_only=True):
        self.results = []
        form = SearchForm(data=self.request.GET)
        if not form.is_valid():
            return None
        self.query = form.cleaned_data['query']

        for model in self.searchables:
            extra = '[kAb]'
            search_field_name = model.get_search_field().name
            qs = model.objects.search(self.query)
            if abundance_only:
                try:
                    model._meta.get_field('abundance')
                except FieldDoesNotExist:
                    abund_a = getattr(model, 'abundance_accessor', None)
                else:
                    abund_a = 'abundance'

                if abund_a:
                    extra = ''
                    qs = qs.exclude(**{abund_a: None})

            hits = [
                (obj, getattr(obj, search_field_name))
                for obj in qs.iterator()
            ]

            if True or hits:
                self.results.append((
                    model._meta.model_name,
                    model._meta.verbose_name_plural + extra,
                    hits,
                ))
