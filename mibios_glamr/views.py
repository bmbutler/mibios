from itertools import chain

from django_tables2 import SingleTableView

from .models import Dataset
from .tables import DatasetTable


class DemoFrontPageView(SingleTableView):
    model = Dataset
    template_name = 'mibios_glamr/demo_frontpage.html'
    table_class = DatasetTable

    def get_table_data(self):
        data = super().get_table_data()
        return chain([Dataset.orphans], data)
