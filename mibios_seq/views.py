from mibios.views import CSVRenderer, CSVRendererZipped
from mibios.views import ExportMixin, TableView, TableViewPlugin
from mibios_seq import models


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


class ExportSharedView(ExportMixin, TableView):
    # Supported export format registry
    # (name, file suffix, http content type, renderer class)
    FORMATS = (
        ('shared/zipped', '.shared.zip', CSVRendererZipped),
        ('shared', '.shared', CSVRenderer),
    )
    DEFAULT_FORMAT = 'shared/zipped'

    def get_filename(self):
        return self.project_name

    def setup(self, request, *args, project, **kwargs):
        self.project_name = project
        super().setup(request, *args, **kwargs)

    def get_values(self):
        return (
            self.get_queryset()
            .filter(project__name=self.project_name)
            .as_shared_values_list()
        )
