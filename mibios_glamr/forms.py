from django import forms


class SearchForm(forms.Form):
    query = forms.CharField(
        strip=True,
        required=True,
        label='',
        help_text='',
    )
