from django.urls import path

from . import views
from .models import Abundance


app_name = 'mibios_seq'

_abundance = Abundance._meta.model_name

urlpatterns = [
    path(
        _abundance + '/export-shared-form/',
        views.ExportSharedFormView.as_view(data_name=_abundance),
        name='export_shared_form',
    ),
    path(
        _abundance + '/avg/export-shared-form/',
        views.ExportAvgSharedFormView.as_view(data_name=_abundance),
        name='export_avg_shared_form',
    ),
    path(
        _abundance + '/export-shared/',
        views.ExportSharedView.as_view(data_name=_abundance),
        name='export_shared',
    ),
    path(
        _abundance + '/avg/export-shared/',
        views.ExportAvgSharedView.as_view(data_name=_abundance),
        name='export_avg_shared',
    ),
    path(
        'sequence/fasta-export/',
        views.ExportSequenceFastaView.as_view(data_name='sequence'),
        name='export_seq_fasta',
    ),
    path(
        'otu/fasta-export/',
        views.ExportOTUFastaView.as_view(data_name='otu'),
        name='export_otu_fasta',
    ),
]
