from django.http import Http404
from django.utils.text import slugify

from mibios.views import CSVRenderer, CSVRendererZipped
from mibios.views import ExportMixin, TableView, TableViewPlugin
from mibios_seq import models


NORMAL_SAMPLE_SIZE = 10000
TYPE_ABS = 'absolute'
TYPE_NORM = f'normalized (to {NORMAL_SAMPLE_SIZE} sample size)'
TYPE_UNIT = 'normalized to [0,1]'


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

        ctx['shared_types'] = [TYPE_ABS, TYPE_NORM, TYPE_UNIT]
        return ctx


class ExportSharedView(ExportMixin, TableView):
    # Supported export format registry
    # (name, file suffix, http content type, renderer class)
    FORMATS = (
        ('shared/zipped', '.shared.zip', CSVRendererZipped),
        ('shared', '.shared', CSVRenderer),
    )
    DEFAULT_FORMAT = 'shared/zipped'

    def get_filename(self):
        return slugify(self.project_name + ' ' + self.shared_type)

    def setup(self, request, *args, project, shared_type, **kwargs):
        if shared_type not in [TYPE_ABS, TYPE_NORM, TYPE_UNIT]:
            raise Http404

        self.project_name = project
        self.shared_type = shared_type
        super().setup(request, *args, **kwargs)

    def get_values(self):
        if self.shared_type == TYPE_ABS:
            norm = None
        elif self.shared_type == TYPE_NORM:
            norm = NORMAL_SAMPLE_SIZE
        elif self.shared_type == TYPE_UNIT:
            norm = 0

        return (
            self.get_queryset()
            .filter(project__name=self.project_name)
            .as_shared_values_list(normalize=norm)
        )
