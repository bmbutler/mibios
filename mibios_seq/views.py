from django.http import Http404
# from django.urls import reverse
from django.utils.text import slugify
from django.views.generic.edit import FormMixin

from mibios.views import CSVRenderer, CSVRendererZipped
from mibios.views import (ExportBaseMixin, ExportMixin, TableView,
                          TableViewPlugin)
from mibios_seq import models

from .forms import ExportSharedForm



class AbundancePlugin(TableViewPlugin):
    model_class = models.Abundance
    template_name = 'mibios_seq/abundance_plugin.html'

    def get_context_data(self, **ctx):
        view = ctx['view']

        # only offer exporting shared data if single project:
        stats = view.get_queryset().get_field_stats('project', natural=True)
        if 'uniform' in stats:
            ctx['uniform_project'] = list(stats['uniform'].keys())[0]
        else:
            ctx['uniform_project'] = None
        return ctx


class ExportSharedFormMixin:
    # Supported export format registry
    # (name, file suffix, http content type, renderer class)
    FORMATS = (
        ('shared/zipped', '.shared.zip', CSVRendererZipped),
        ('shared', '.shared', CSVRenderer),
    )
    DEFAULT_FORMAT = 'shared/zipped'

    group_col_choice_map = {
        'sequencing record': ('sequencing__name', ),
        'sample ID': ('sequencing__sample__fecalsample__natural', ),
        # (('participant__name', 'week__number', 'participant and week'),
    }

    @property
    def group_col_choices(self):
        """
        Supply the choices for the group_col form field
        """
        return [(slugify(i), i) for i in self.group_col_choice_map.keys()]

    def get_form_class(self):
        return ExportSharedForm.factory(self)


class ExportSharedFormView(ExportSharedFormMixin, ExportBaseMixin, FormMixin,
                           TableView):
    template_name = 'mibios/export.html'
    export_form_action_url = '/abundance/export-shared/'

    def get_queryset(self):
        # no need for the data, makes us speedy
        # FIXME: TableView should probably be split up more into a mixin that
        # contains all the table stuff except the data and a view that adds
        # just the table data, so we don't have to make the empty QuerySet
        # here.
        return self.model.objects.filter(pk=None)

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['export_form_action_url'] = self.export_form_action_url
        return ctx


class ExportSharedView(ExportSharedFormMixin, ExportMixin, TableView):
    _group_id_map = None

    def get_filename(self):
        parts = [self.project_name]
        if self.normalize is None:
            norm = 'abs'
        elif self.normalize == 0:
            norm = 'rel'
        else:
            norm = str(self.normalize)
        parts.append(norm)
        #  TODO:
        #  if <average>:
        #      parts.append('avg')
        return slugify(' '.join(parts))

    def get(self, request, *args, **kwargs):
        form = self.get_form_class()(data=dict(request.GET.items()))
        if not form.is_valid():
            log.debug(f'export shared form invalid:\n{form.errors}')
            raise Http404(form.errors)

        self.project_name = form.cleaned_data['project']
        self.normalize = form.cleaned_data['normalize']
        for k, v in self.group_col_choice_map.items():
            if slugify(k) == form.cleaned_data['group_cols']:
                self.group_id_accessors = \
                        self.model.resolve_natural_lookups(*v)
                self.group_cols_verbose = (k, )
                break

        self.mothur = form.cleaned_data['mothur']
        return super().get(request, *args, **kwargs)

    def _group_id_mapper(self, sequencing_id):
        """
        Helper to map sequencing ids to group column values

        Method returns a tuple of identifying values.
        """
        # FIXME: this is all a bit hhcd/Fecalsample specific
        if self._group_id_map is None:
            a = [i[len('sequencing__'):] for i in self.group_id_accessors]
            m = {}
            # TODO: obey curation status
            qs = models.Sequencing.objects.values_list('pk', 'name', *a)
            for pk, name, *vals in qs:
                if None in vals:
                    m[pk] = name
                else:
                    m[pk] = '_'.join((str(i) for i in vals))
            self._group_id_map = m

        return (self._group_id_map[sequencing_id], )

    def get_values(self):
        return (
            self.get_queryset()
            .filter(project__name=self.project_name)
            .as_shared_values_list(
                self.normalize,
                self._group_id_mapper,
                self.group_cols_verbose,
                mothur=self.mothur,
            )
        )
