from itertools import chain

from django_tables2 import Column, SingleTableView, TemplateColumn
import pandas

from django.conf import settings
from django.contrib import messages
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.core.management import call_command
from django.db.models import Count, URLField
from django.http import Http404, HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.generic import DetailView
from django.views.generic.base import TemplateView

from mibios import get_registry
from mibios.data import TableConfig
from mibios.models import Q
from mibios.views import ExportBaseMixin, TextRendererZipped
from mibios_omics import get_sample_model
from mibios_omics.models import (
    CompoundAbundance, FuncAbundance, TaxonAbundance
)
from mibios_umrad.models import (
    CompoundEntry, CompoundName, FunctionName, FuncRefDBEntry,
)
from mibios_omics.models import Gene
from . import models, tables
from .forms import (
    SearchForm, QBuilderForm, QBuilderAddForm,
)
from .search_utils import searchable_models, get_suggestions


class ExportMixin(ExportBaseMixin):
    query_param = 'export'

    def get_filename(self):
        value = ''
        if hasattr(self, 'object_model'):
            value += self.object_model._meta.model_name
        if hasattr(self, 'object'):
            if value:
                value += '-'
            value += str(self.object)
        if hasattr(self, 'model'):
            if value:
                value += '-'
            value += self.model._meta.model_name
        if value:
            return value
        else:
            return self.__class__.__name__.lower() + '-export'

    def get(self, request, *args, **kwargs):
        if self.export_check():
            return self.export_response()
        else:
            return super().get(request, *args, **kwargs)

    def export_check(self):
        """ Returns wether a file export response is needed """
        return self.query_param in self.request.GET

    def export_response(self):
        """ generate file download response """
        name, suffix, renderer_class = self.get_format()

        response = HttpResponse(content_type=renderer_class.content_type)
        filename = self.get_filename() + suffix
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        renderer_class(response, filename=filename).render(self.get_values())
        return response

    def get_values(self):
        if hasattr(self, 'get_table'):
            return self.get_table().as_values()
        else:
            raise RuntimeError('not implemented')


class BaseFilterMixin:
    """
    Basic filter infrastructure, sufficient to view the filter
    """
    def setup(self, request, *args, model=None, **kwargs):
        super().setup(request, *args, **kwargs)
        self.filter_item_form = None

        if model:
            try:
                self.model = get_registry().models[model]
            except KeyError as e:
                raise Http404(f'no such model: {e}') from e
        self.model_name = self.model._meta.model_name
        if 'search_filter' in request.session \
                and self.model_name == request.session.get('search_model'):
            self.q = Q.deserialize(request.session['search_filter'])
        else:
            # (a) no filter in session yet or
            # (b) data category changed -> so ignore session filter
            self.q = Q()

    def get_queryset(self):
        return super().get_queryset().filter(self.q).distinct()

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['model_name'] = self.model._meta.model_name
        ctx['model_verbose_name'] = self.model._meta.verbose_name
        ctx['model_verbose_name_plural'] = self.model._meta.verbose_name_plural
        ctx['edit'] = False
        ctx['qnode'] = self.q
        ctx['col_width'] = 9
        return ctx


class EditFilterMixin(BaseFilterMixin):
    """
    Provide complex filter editor
    """
    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.old_q = self.q  # keep backup for revert on error

    def post(self, request, *args, **kwargs):
        form = QBuilderForm(data=request.POST)
        if not form.is_valid():
            raise Http404('post request form invalid', form.errors)

        path = form.cleaned_data['path']

        action = request.POST.get('action', None)
        try:
            if action == 'rm':
                self.q = self.q.remove_node(path)
            elif action == 'neg':
                self.q = self.q.negate_node(path)
            elif action == 'add':
                self.filter_item_form = QBuilderAddForm(
                    model=self.model,
                    path=path,
                )
            elif action == 'edit':
                ...
            elif action == 'apply_change':
                self.q = self.apply_changes()
        except IndexError:
            # illegal path, e.g. remove and then resend POST
            # so ignoring this
            # TODO: review error handling here
            pass

        try:
            self.model.objects.filter(self.q)
        except Exception as e:
            self.q = self.old_q  # revert changes
            messages.add_message(
                request,
                messages.ERROR,
                f'not good: {e.__class__.__name__}: {e}',
            )
        else:
            request.session['search_filter'] = self.q.serialize()
            request.session['search_model'] = self.model_name
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['edit'] = True
        ctx['qnode_path'] = None
        return ctx

    def apply_changes(self):
        form = QBuilderAddForm(
            model=self.model,
            data=self.request.POST,
        )
        if not form.is_valid():
            raise Http404('apply changes request form invalid', form.errors)
        lhs = form.cleaned_data['key']
        lhs += '__' + form.cleaned_data['lookup']
        rhs = form.cleaned_data['value']
        path = form.cleaned_data['path']
        return self.q.combine_at_node(Q(**{lhs: rhs}), path)


class ModelTableMixin(ExportMixin):
    """
    Mixin for SingleTableView

    Improves columns for relation fields.  The inheriting view must set
    self.model
    """
    model = None  # model needs to be set by inheriting class
    table_class = None  # triggers the model-based table class creation
    exclude = ['id']  # do not display these fields

    def get_table_kwargs(self):
        kw = {}
        kw['exclude'] = self.exclude
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
            elif i.name in self.exclude:
                continue
            else:
                # regular FK field
                col = Column(
                    linkify=lambda value: tables.get_record_url(value)
                )
            cols.append((i.name, col))
        return cols


class AbundanceView(ExportMixin, SingleTableView):
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
            self.object_model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.object_model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

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


class AbundanceGeneView(ModelTableMixin, SingleTableView):
    """
    Views genes for a sample/something combo

    Can export genes in fasta format
    """
    template_name = 'mibios_glamr/abundance_genes.html'
    model = Gene
    exclude = ['id', 'sample']

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.object_model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.object_model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

        try:
            self.sample = models.Sample.objects.get(accession=kwargs['sample'])
        except models.Sample.DoesNotExist:
            raise Http404('no such sample')

    def get_queryset(self):
        f = dict(sample=self.sample)
        if self.object_model is FuncRefDBEntry:
            f['besthit__function_refs'] = self.object
        return Gene.objects.filter(**f)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['sample'] = self.sample
        ctx['object'] = self.object
        ctx['object_model_name'] = self.object_model._meta.model_name
        ctx['object_model_name_verbose'] = self.object_model._meta.verbose_name
        return ctx

    # file export methods

    def export_check(self):
        self.export_fasta = 'export-fasta' in self.request.GET
        return self.export_fasta or super().export_check()

    def get_format(self):
        return ('fa/zip', '.fasta.zip', TextRendererZipped)

    def get_filename(self):
        if self.export_fasta:
            return super().get_filename() + '.fasta'
        else:
            return super().get_filename()

    def get_values(self):
        if self.export_fasta:
            return self.get_queryset().to_fasta()
        else:
            return super().get_values()


class BaseDetailView(DetailView):
    template_name = 'mibios_glamr/detail.html'
    max_to_many = 16

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['object_model_name'] = self.model._meta.model_name
        ctx['object_model_verbose_name'] = self.model._meta.verbose_name
        ctx['details'], ctx['relations'] = self.get_details()
        ctx['external_url'] = self.object.get_external_url()

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

        if exturl := self.object.get_external_url():
            details.append(('external URL', exturl, exturl))

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
        orphans = models.Dataset.orphans
        orphans.sample_count = orphans.samples().count()
        # put orphans into first row (if any exist):
        if orphans.sample_count > 0:
            return chain([orphans], data)
        else:
            return data

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.select_related('reference')
        qs = qs.annotate(sample_count=Count('sample'))
        return qs

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['mc_abund'] = TaxonAbundance.objects \
            .filter(taxname='MICROCYSTIS') \
            .select_related('sample')[:5]
        ctx['keyword_search_form'] = SearchForm()
        self.make_ratios_plot()
        return ctx

    def make_ratios_plot(self):
        imgpath = settings.MEDIA_ROOT + 'var/mappedratios.png'
        ratios = pandas.DataFrame([
            (i.reads_mapped_contigs / i.read_count,
             i.reads_mapped_genes / i.read_count)
            for i in get_sample_model().objects.all()
            if i.contigs_ok and i.genes_ok
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


class OverView(SingleTableView):
    template_name = 'mibios_glamr/overview.html'
    table_class = tables.OverViewTable

    # lookup from sample to object
    accessor = {
        'compoundentry': 'compoundabundance__compound',
        'funcrefdbentry': 'funcabundance__function',
        'taxname': 'taxonabundance__taxname',
        'compoundname': 'compoundabundance__compound__names',
        'functionname': 'funcabundance__function__names',
    }

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

    def get_table_data(self):
        try:
            a = self.accessor[self.model._meta.model_name]
        except KeyError:
            return []
        else:
            a = 'sample__' + a
            f = {a: self.object}
            qs = models.Dataset.objects.filter(**f)
            qs = qs.annotate(num_samples=Count('sample', distinct=True))

        # add totals (can't do this in above query (cf. django docs order of
        # annotation and filters:
        totals = models.Dataset.objects.filter(pk__in=[i.pk for i in qs])
        totals = totals.annotate(Count('sample'))
        totals = dict(totals.values_list('pk', 'sample__count'))
        for i in qs:
            i.total_samples = totals[i.pk]

        return qs

    def get_table(self):
        table = super().get_table()
        # FIXME: a hack: pass some extra context for column rendering
        table.view_object = self.object
        table.view_object_model_name = self.model._meta.model_name
        return table

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx.update(
            object=self.object,
            object_model_name=self.model._meta.model_name,
            object_model_name_verbose=self.model._meta.verbose_name
        )
        return ctx


class OverViewSamplesView(SingleTableView):
    template_name = 'mibios_glamr/overview_samples.html'
    table_class = tables.OverViewSamplesTable

    # lookup from sample to object
    accessor = {
        'compoundentry': 'compoundabundance__compound',
        'funcrefdbentry': 'funcabundance__function',
        'taxname': 'taxonabundance__taxname',
        'compoundname': 'compoundabundance__compound__names',
        'functionname': 'funcabundance__function__names',
    }

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        try:
            self.model = get_registry().models[kwargs['model']]
        except KeyError as e:
            raise Http404(f'no such model: {e}') from e

        try:
            self.object = self.model.objects.get(pk=kwargs['pk'])
        except self.model.DoesNotExist:
            raise Http404('no such object')

    def get_table_data(self):
        try:
            a = self.accessor[self.model._meta.model_name]
        except KeyError:
            return []
        else:
            f = {a: self.object}
            qs = models.Sample.objects.filter(**f)
            # distinct: there may be multiple abundances per object and sample
            # e.g. for different roles of a compound
            qs = qs.select_related('group').distinct()
        return qs

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx.update(
            object=self.object,
            object_model_name=self.model._meta.model_name,
            object_model_name_verbose=self.model._meta.verbose_name
        )
        return ctx


class SampleListView(SingleTableView):
    """ List of samples belonging to a given dataset  """
    model = get_sample_model()
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
    model = get_sample_model()


class SearchView(TemplateView):
    """ offer a form for advanced search, offer model list """
    template_name = 'mibios_glamr/search_init.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['keyword_search_form'] = SearchForm()
        ctx['models'] = [
            (i._meta.model_name, i._meta.verbose_name)
            for i in get_registry().models.values()
        ]
        return ctx


class SearchModelView(EditFilterMixin, TemplateView):
    """ offer model-based searching """
    template_name = 'mibios_glamr/search_model.html'
    model = None

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['keyword_search_form'] = SearchForm()
        return ctx


class SearchHitView(TemplateView):
    template_name = 'mibios_glamr/search_hits.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['search_hits'] = self.hits
        ctx['query'] = self.query
        ctx['no_hit_models'] = self.no_hit_models
        ctx['suggestions'] = self.suggestions
        return ctx

    def get(self, request, *args, **kwargs):
        self.search()
        if self.hits:
            self.suggestions = None
        else:
            self.suggestions = get_suggestions(self.query)
            if not self.suggestions:
                messages.add_message(
                    request, messages.INFO, 'search: did not find anything'
                )
                return HttpResponseRedirect(reverse('frontpage'))
        return self.render_to_response(self.get_context_data())

    def get_suggestions(self):
        return None

    def search(self):
        self.hits = []
        self.no_hit_models = []

        form = SearchForm(data=self.request.GET)
        if not form.is_valid():
            return None
        self.query = form.cleaned_data['query']

        for model in searchable_models:
            have_abundance = False
            search_field_name = model.get_search_field().name
            qs = model.objects.search(self.query)

            if not form.cleaned_data.get('search_all'):
                # just search for objects with abundance
                # (if model supports it, no filters for other models)
                if model is CompoundName:
                    abund_lookup = 'compoundentry__abundance'
                elif model is FunctionName:
                    abund_lookup = 'funcrefdbentry__abundance'
                else:
                    abund_lookup = 'abundance'
                try:
                    qs = qs.exclude(**{abund_lookup: None})
                except FieldError:
                    # don't have abundance field, keeping qs as-is
                    # (or abund_path is outdated)
                    continue
                else:
                    have_abundance = True

            model_hits = []
            if model is CompoundName:
                model_name = CompoundEntry._meta.model_name
                for obj in qs.iterator():
                    hit_value = getattr(obj, search_field_name)
                    for i in obj.compoundentry_set.exclude(abundance=None):
                        model_hits.append((i, str(i), hit_value))
            elif model is FunctionName:
                model_name = FuncRefDBEntry._meta.model_name
                for obj in qs.iterator():
                    hit_value = getattr(obj, search_field_name)
                    for i in obj.funcrefdbentry_set.exclude(abundance=None):
                        model_hits.append((i, str(i), hit_value))
            else:
                model_name = model._meta.model_name
                for obj in qs.iterator():
                    # use hit as-is, no proxy
                    hit_value = getattr(obj, search_field_name)
                    model_hits.append((obj, hit_value, None))

            if model_hits:
                self.hits.append((
                    have_abundance,
                    model._meta.verbose_name_plural,
                    model_name,
                    model_hits,
                ))
            else:
                self.no_hit_models.append(model._meta.verbose_name_plural)


class TableView(BaseFilterMixin, ModelTableMixin, SingleTableView):
    template_name = 'mibios_glamr/table.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        self.conf = TableConfig(self.model)


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


class ToManyFullListView(ModelTableMixin, ToManyListView):
    """ relations view but with full model-based table """
    template_name = 'mibios_glamr/relations_full_list.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        # hide the column for the object
        self.exclude.append(self.field.remote_field.name)
