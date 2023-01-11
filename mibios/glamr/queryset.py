from collections import Counter

from django.db.models import Count

import pandas

from mibios.umrad.manager import QuerySet


class DatasetQuerySet(QuerySet):
    def basic_counts(
            self,
            column_field='sample_type',
            row_field='geo_loc_name',
            exclude_blanks=True,
            as_dataframe=True,
    ):
        """
        Compute basic statistics for front page display

        Parameters:  column_field and row_fields must be char field names from
        the Sample model.

        Datasets are counted multiples times if they contain samples from
        different lakes or of different type.
        """
        our_col_field = 'sample__' + column_field
        our_row_field = 'sample__' + row_field
        qs = (self
              .values_list('pk', our_row_field, our_col_field)
              .order_by(our_row_field, our_col_field)
              .distinct()
              )

        counts = Counter(((a, b) for _, a, b in qs if a and b))
        if not as_dataframe:
            return counts

        counts = pandas.Series(
            counts.values(),
            index=pandas.MultiIndex.from_tuples(
                counts.keys(),
                names=(row_field, column_field),
            ),
        )
        return counts.unstack(fill_value=0)


class SampleQuerySet(QuerySet):
    def basic_counts(
            self,
            column_field='sample_type',
            row_field='geo_loc_name',
            exclude_blanks=True,
            as_dataframe=True,
    ):
        """
        Compute basic statistics for front page display
        """
        qs = self
        if exclude_blanks:
            qs = qs.exclude(**{column_field: ''}).exclude(**{row_field: ''})
        qs = (qs
              .values_list(row_field, column_field)
              .order_by(row_field, column_field)
              .annotate(count=Count('*'))
              # .order_by('-count')
              )

        if not as_dataframe:
            return qs

        # It's not totally trivial to turn the QuerySet, a sequence of tuples
        # (row_name, column_name, count) into a DataFrame.  So we put the
        # counts into a Series and make a multi-index from the
        # row/column-names.  Then we need to pivot this, but keep it integer
        # and fill missing values with zero (which unstack() can do with a
        # multi index)
        counts = pandas.Series(
            (count for _, _, count in qs),
            index=pandas.MultiIndex.from_tuples(
                ((a, b) for a, b, _ in qs),
                names=(row_field, column_field),
            ),
        )
        return counts.unstack(fill_value=0)
