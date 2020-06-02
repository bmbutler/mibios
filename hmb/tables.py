import django_tables2 as tables
from .models import Sequencing


class SequencingTable(tables.Table):
    class Meta:
        model = Sequencing
        template_name = 'django_tables2/bootstrap.html'
        fields = (
            'name', 'sample.participant', 'sample', 'sample.week',
            'sample.participant.semester', 'batch', 'note',
        )
