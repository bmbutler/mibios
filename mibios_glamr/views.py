from itertools import chain

from django_tables2 import Column, SingleTableView, TemplateColumn
import pandas

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.core.management import call_command
from django.db.models import URLField
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse
from django.views.generic import DetailView
from django.views.generic.base import TemplateView

from mibios import get_registry
from mibios_omics.models import (
    CompoundAbundance, FuncAbundance, Sample, TaxonAbundance
)
from mibios_umrad.models import (
    CompoundEntry, CompoundName, FunctionName, Location, Metal, FuncRefDBEntry,
    ReactionEntry, TaxName, Taxon, Uniprot, UniRef100,
)
from . import models
from .forms import SearchForm
from . import tables


class AbundanceView(SingleTableView):
    """
    Lists abundance data for a single object of a variable model
    """
    template_name = 'mibios_glamr/abundance.html'

    def get_table_class(self):
        if self.model is CompoundAbundance:
            return tables.CompoundAbundanceTable
        elif self.model is FuncAbundance:
            return tables.FunctionAbundanceTable
        elif self.model is TaxonAbundance:
            return tables.TaxonAbundanceTable
        else:
            raise ValueError('unsupported abundance model')

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            obj_model = get_registry().models[self.kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        accession = self.request.GET['id'].strip()

        try:
            sfield = obj_model.get_search_field()
        except AttributeError as e:
            raise Http404('model is unsupported') from e

        kw = {sfield.name: accession}
        self.object = obj_model.objects.get(**kw)
        self.object_model = obj_model
        self.model = self.object.abundance.model

    def get_queryset(self):
        try:
            return self.object.abundance.all()
        except AttributeError:
            # (object-)model lacks reverse abundance relation
            raise

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_name_verbose'] = self.model._meta.verbose_name
        ctx['object'] = self.object
        ctx['object_model_name'] = self.object_model._meta.model_name

        return ctx


class BaseDetailView(DetailView):
    template_name = 'mibios_glamr/detail.html'
    max_to_many = 16

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object_model_name'] = self.model._meta.model_name
        ctx['object_model_verbose_name'] = self.model._meta.verbose_name
        ctx['details'], ctx['relations'] = self.get_details()
        return ctx

    def get_details(self):
        details = []
        rel_lists = []
        for i in self.model._meta.get_fields():
            if i.name == 'id':
                continue

            # some relations (e.g.: 1-1) don't have a verbose name:
            name = getattr(i, 'verbose_name', i.name)

            if i.many_to_many or i.one_to_many:
                model_name = i.related_model._meta.model_name
                try:
                    # trying as m2m relation (other side of declared field)
                    rel_attr = i.get_accessor_name()
                except AttributeError:
                    # this is the m2m field
                    rel_attr = i.name
                qs = getattr(self.object, rel_attr).all()[:self.max_to_many]
                rel_lists.append((name, model_name, qs, i))
                continue

            value = getattr(self.object, i.name, None)
            if value:
                if i.many_to_one or i.one_to_one:  # TODO: test 1-1 fields
                    url = tables.get_record_url(value)
                elif isinstance(i, URLField):
                    url = value
                else:
                    url = None
            else:
                url = None

            if hasattr(i, 'choices') and i.choices:
                value = getattr(self.object, f'get_{i.name}_display')()

            details.append((name, url, value))
        return details, rel_lists


class DatasetView(BaseDetailView):
    model = models.Dataset

    def get_object(self):
        if self.kwargs.get(self.pk_url_kwarg) == 0:
            return models.Dataset.orphans
        return super().get_object()

    def get_details(self):
        details, rel_lists = super().get_details()
        if self.object.orphan_group:
            pk = 0
        else:
            pk = self.object.pk
        # prepend samples row
        details = [(
            'Samples',
            reverse('dataset_sample_list', args=[pk]),
            f'List of {self.object.samples().count()} samples'
        )] + details
        return details, rel_lists


class DemoFrontPageView(SingleTableView):
    model = models.Dataset
    template_name = 'mibios_glamr/demo_frontpage.html'
    table_class = tables.DatasetTable

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


class ReferenceView(BaseDetailView):
    model = models.Reference


class RecordView(BaseDetailView):
    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e


class SampleListView(SingleTableView):
    """ List of samples belonging to a given dataset  """
    model = Sample
    template_name = 'mibios_glamr/sample_list.html'
    table_class = tables.SampleTable

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


class SampleView(BaseDetailView):
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

    def search(self):
        self.results = []
        form = SearchForm(data=self.request.GET)
        if not form.is_valid():
            return None
        self.query = form.cleaned_data['query']

        for model in self.searchables:
            have_abundance = False
            search_field_name = model.get_search_field().name
            qs = model.objects.search(self.query)

            try:
                abund_qs = qs.exclude(abundance=None)
            except FieldError:
                # don't have abundance field, keeping qs as-is
                pass
            else:
                have_abundance = True
                if not form.cleaned_data['search_all']:
                    # just search for objects with abundance
                    # (if model supports it, no filters for other models)
                    qs = abund_qs

            hits = [
                (obj, getattr(obj, search_field_name))
                for obj in qs.iterator()
            ]

            if True or hits:
                self.results.append((
                    have_abundance,
                    model._meta.model_name,
                    model._meta.verbose_name_plural,
                    hits,
                ))


class ToManyListView(SingleTableView):
    """ view relations belonging to one object """
    template_name = 'mibios_glamr/relations_list.html'
    table_class = tables.SingleColumnRelatedTable

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.obj_model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.obj_model.objects.get(pk=kwargs['pk'])
        except self.obj_model.DoesNotExist as e:
            raise Http404('no such record') from e

        try:
            field = self.obj_model._meta.get_field(kwargs['field'])
        except FieldDoesNotExist as e:
            raise Http404('no such field') from e

        if not field.one_to_many and not field.many_to_many:
            raise Http404('field is not *-to_many')

        self.field = field
        self.model = field.related_model

        try:
            self.accessor_name = field.get_accessor_name()
        except AttributeError:
            self.accessor_name = field.name

    def get_queryset(self):
        return getattr(self.object, self.accessor_name).all()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object'] = self.object
        ctx['object_model_name'] = self.obj_model._meta.model_name
        ctx['object_model_name_verbose'] = self.obj_model._meta.verbose_name
        ctx['field'] = self.field
        ctx['model_name_verbose'] = self.model._meta.verbose_name
        return ctx


class ToManyFullListView(ToManyListView):
    """ relations view but with full tables """
    template_name = 'mibios_glamr/relations_full_list.html'
    table_class = None

    def get_table_kwargs(self):
        kw = {}
        # excl. pk and column for the object
        kw['exclude'] = ['id', self.field.remote_field.name]
        kw['extra_columns'] = self.get_improved_columns()
        return kw

    def get_improved_columns(self):
        """ make replacements to linkify FK + accession columns """
        cols = []
        try:
            acc_field = self.model.get_accession_field_single()
        except RuntimeError:
            acc_field = None
            col = TemplateColumn(
                """[<a href="{% url 'record' model=model_name pk=record.pk %}">{{ record }}</a>]""",  # noqa:E501
                extra_context=dict(model_name=self.model._meta.model_name),
            )
            cols.append(('record links', col))

        for i in self.model._meta.get_fields():
            if acc_field and i is acc_field:
                col = Column(
                    linkify=lambda record: tables.get_record_url(record)
                )
            elif not i.many_to_one:
                continue
            elif i is self.field:
                # is excluded
                continue
            else:
                # regular FK field
                col = Column(
                    linkify=lambda value: tables.get_record_url(value)
                )
            cols.append((i.name, col))
        return cols
