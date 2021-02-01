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

    meta_col_choice_map = {
        # TODO: implement auto-detection
        'sequencing record ID': 'sequencing__name',
    }

    norm_choices = (
        (ExportSharedForm.NORM_NONE, 'none'),
        (0, 'fractions'),
        (10000, 'to 10000')
    )

    @property
    def meta_col_choices(self):
        """
        Supply the choices for the meta_col form field
        """
        return [(slugify(i), i) for i in self.meta_col_choice_map.keys()]

    def get_form_class(self):
        return ExportSharedForm.factory(self)


class ExportSharedFormView(ExportSharedFormMixin, ExportBaseMixin,
                           DatasetMixin, FormView):
    template_name = 'mibios/export.html'
    export_url_name = 'mibios_seq:export_shared'

    def get_context_data(self, **ctx):
        ctx = super().get_context_data(**ctx)
        ctx['export_data_name'] = '"shared" abundance'
        return ctx


class ExportAvgSharedFormView(AverageMixin, ExportSharedFormView):
    export_url_name = 'mibios_seq:export_avg_shared'

    def get_form_class(self):
        if self.avg_by:
            # override meta col choices: meta is avg_by but without project
            # nor otu since those are always averaged by.  What we want is e.g.
            # just participant and week:
            # TODO: auto-detect other relations
            self.meta_col_choice_map = dict()
            for i in self.avg_by:
                if i in ['project', 'otu']:
                    continue
                model_name = i.split('__')[-1]
                self.meta_col_choice_map[model_name.capitalize()] = \
                    model_name + '__natural'

            # override norm choices: rm "none"
            # as there's no such thing as average absolute counts
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
        form = self.get_form_class()(data=request.GET)
        self.form = form
        if not form.is_valid():
            log.debug(f'export shared form invalid:\n{form.errors}')
            raise Http404(form.errors)

        log.debug(f'shared export form valid: {form.cleaned_data}')
        self.project_name = form.cleaned_data['project']
        self.normalize = form.cleaned_data['normalize']
        self.meta_col_accessors = []
        for i in form.cleaned_data['meta_cols']:
            for k, v in self.meta_col_choice_map.items():
                if i == slugify(k):
                    self.meta_col_accessors.append(v)
                    break
            else:
                raise ValueError(
                    'failed to map form-chosen column {i} to accessor'
                )

        self.mothur = form.cleaned_data['mothur']
        return super().get(request, *args, **kwargs)

    def get_values(self):
        return (
            self.get_queryset()
            .filter_project(self.project_name)
            .as_shared_values_list(
                self.normalize,
                meta_cols=self.meta_col_accessors,
                mothur=self.mothur,
            )
        )


class ExportAvgSharedView(ExportAvgSharedFormView, ExportSharedView):
    def get_filename(self):
        return super().get_filename() + '-avg'

