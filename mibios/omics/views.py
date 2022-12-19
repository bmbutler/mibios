from django.http import Http404, HttpResponse

from . import get_sample_model
from .models import TaxonAbundance


def krona(request, sample_pk):
    """
    Display Krona visualization for taxon abundance of one sample
    """
    Sample = get_sample_model()
    try:
        sample = Sample.objects.get(pk=sample_pk)
    except Sample.DoesNotExist:
        raise Http404('no such sample')

    html = TaxonAbundance.objects.as_krona_html(sample, 'sum_gene_rpkm')
    if html is None:
        return HttpResponse(html)
    else:
        raise Http404('no abundance data for sample or error with krona')
