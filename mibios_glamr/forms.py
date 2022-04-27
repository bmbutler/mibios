from django import forms


class SearchForm(forms.Form):
    query = forms.CharField(
        strip=True,
        required=True,
        label='',
        help_text='',
    )
    search_all = forms.BooleanField(
        initial=False,
        required=False,
        # label=??,
        # help_text='Also show results for things not found in samples',
    )


class AdvancedSearchForm(SearchForm):
    pass
