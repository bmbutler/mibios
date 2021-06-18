from .models import Contig, Gene


def populate():
    """
    populate DB for testing
    """
    c1 = Contig.objects.create(accession='c1')
    c2 = Contig.objects.create(accession='c2')
    c3 = Contig.objects.create(accession='c3')
    g1 = Gene.objects.create(accession='g1', contig=c1, start=1, end=4, strand=Gene.STRAND_PLUS)  # noqa:E501
    g2 = Gene.objects.create(accession='g2', contig=c1, start=5, end=9, strand=Gene.STRAND_PLUS)  # noqa:E501
    g3 = Gene.objects.create(accession='g3', contig=c3, start=1, end=7, strand=Gene.STRAND_MINUS)  # noqa:E501

    # g1.add_dist(g2, 1.234)
    # g1.add_dist(g3, 1.99)

    return (c1, c2, c3), (g1, g2, g3)


def clean():
    return Contig.objects.all().delete()
