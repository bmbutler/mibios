from django.http import Http404
from django.http.request import QueryDict
from django.utils.text import slugify
from django.views.generic.edit import FormView

from mibios import QUERY_AVG_BY
from mibios.utils import getLogger
from mibios.views import AverageMixin, CSVTabRenderer, CSVTabRendererZipped
from mibios.views import (DatasetMixin, ExportBaseMixin, ExportMixin,
                          TableView, TableViewPlugin)
from mibios_seq import models

from .forms import ExportSharedForm

log = getLogger(__name__)


class AbundancePlugin(TableViewPlugin):
    model_class = models.Abundance
    template_name = 'mibios_seq/abundance_plugin.html'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        if self.view.avg_by:
            # Normally TableView with AverageMixin gets the avg-by fields from
            # the URL path, but for the shared export view we put them into the
            # query string.  The idea is that eventually will completely switch
            # to avg-by in query string.
            # add avg-by spec to query string:
            qavg = QueryDict(mutable=True)
            qavg.setlist(QUERY_AVG_BY, self.view.avg_by)
            q = ctx['querystr']
            if q:
                q += '&'
            else:
                q = '?'
            q += qavg.urlencode()
            ctx['querystr'] = q
        return ctx


class ExportSharedFormMixin:
    # Supported export format registry
    # (name, file suffix, http content type, renderer class)
    FORMATS = (
        ('shared/zipped', '.shared.zip', CSVTabRendererZipped),
        ('shared', '.shared', CSVTabRenderer),
    )
    DEFAULT_FORMAT = 'shared/zipped'

    group_col_choice_map = {
        'sequencing record': ('sequencing__name', ),
        'sample ID': ('sequencing__sample__fecalsample__natural', ),
        # (('participant__name', 'week__number', 'participant and week'),
    }

    norm_choices = (
        (ExportSharedForm.NORM_NONE, 'none'),
        (0, 'fractions'),
        (10000, 'to 10000')
    )

    @property
    def group_col_choices(self):
        """
        Supply the choices for the group_col form field
        """
        return [(slugify(i), i) for i in self.group_col_choice_map.keys()]

    def get_form_class(self):
        return ExportSharedForm.factory(self)


class ExportSharedFormView(ExportSharedFormMixin, ExportBaseMixin,
                           DatasetMixin, FormView):
    template_name = 'mibios/export.html'
    export_form_action_url = '/abundance/export-shared/'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['export_form_action_url'] = self.export_form_action_url
        ctx['export_data_name'] = '"shared" abundance'
        return ctx


class ExportAvgSharedFormView(AverageMixin, ExportSharedFormView):
    export_form_action_url = '/abundance/avg/export-shared/'

    def get_form_class(self):
        if self.avg_by:
            # override group col choices: group is avg_by but without project
            # and otu since those are always averaged by.  What we want is e.g.
            # just participant and week:
            bys = [i for i in self.avg_by if i not in ['project', 'otu']]
            # combine last lookup components for label (will become the shared
            # file group column header):
            label = '_'.join([i.rpartition('__')[2] for i in bys])
            self.group_col_choice_map = {label: bys}

            # override norm choices: rm "none"
            self.norm_choices = tuple(self.norm_choices[1:])

        return super().get_form_class()


class ExportSharedView(ExportSharedFormMixin, ExportMixin, TableView):
    _group_id_maps = None

    def get_filename(self):
        parts = [self.project_name]
        if self.normalize is None:
            norm = 'abs'
        elif self.normalize == 0:
            norm = 'rel'
        else:
            norm = str(self.normalize)
        parts.append(norm)
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
        # FIXME: can this be unified/generalized with the avg version?
        if self._group_id_maps is None:
            a = [i[len('sequencing__'):] for i in self.group_id_accessors]
            m = {}
            # TODO: obey curation status
            qs = models.Sequencing.objects.values_list('pk', 'name', *a)
            for pk, name, *vals in qs:
                if None in vals:
                    m[pk] = name
                else:
                    m[pk] = '_'.join((str(i) for i in vals))
            self._group_id_maps = [m]

        return (self._group_id_maps[0][sequencing_id], )

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


class ExportAvgSharedView(ExportAvgSharedFormView, ExportSharedView):
    def get_filename(self):
        return super().get_filename() + '-avg'

    def _group_id_mapper(self, *row_ids):
        """
        Helper to map pk to natural key

        Method returns a tuple of identifying values.
        """
        if self._group_id_maps is None:
            self._group_id_maps = []
            for a in self.avg_by:
                if a in ['project', 'otu']:
                    continue
                f = self.model.get_field(a)
                m = {}
                # TODO: obey curation status
                for i in f.related_model.objects.all():
                    m[i.pk] = i.natural

                self._group_id_maps.append(m)

        return (
            j[i]
            for i, j in zip(row_ids, self._group_id_maps)
        )

    def get_values(self):
        # FIXME: this is hackish, should give e.g. ['participant', 'week']
        # for two columns
        self.group_cols_verbose = [
            i.rpartition('__')[2]
            for i in self.avg_by
            if i not in ['project', 'otu']
        ]
        return super().get_values()
