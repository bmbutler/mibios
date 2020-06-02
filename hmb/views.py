from django.apps import apps
from django_tables2 import SingleTableView, Table


class ModelIndexView(SingleTableView):
    # gets set via as_view(model=somemodel)
    model = None
    template_name = 'hmb/model_index.html'

    def get_queryset(self):
        if self.model is None:
            return []
        else:
            return super().get_queryset()

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
    def get_model_urls(cls):
        ret = []
        for i in apps.get_app_config('hmb').get_models():
            m = i._meta.model_name
            ret.append(('hmb:{}_index'.format(m), m))
        return sorted(ret)

    def get_context_data(self, **ctx):
        ctx.update(super().get_context_data())
        if self.model is not None:
            ctx['model_name'] = self.model._meta.verbose_name
        ctx['model_urls'] = self.get_model_urls()
        return ctx
