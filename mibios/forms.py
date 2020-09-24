from django import forms

from . import QUERY_FILTER, QUERY_FIELD, QUERY_FORMAT


class UploadFileForm(forms.Form):
    file = forms.FileField()
    dry_run = forms.BooleanField(
        required=False,
        initial=False,
        label='Dry-run',
        help_text='goes through the whole import process but changes are '
                  'not committed to database',
    )
    overwrite = forms.BooleanField(
        widget=forms.RadioSelect(choices=((True, 'overwrite'), (False, 'append-only'))),
        initial=False,
        required=False,
        label='data import mode',
        help_text='whether to allow overwriting non-empty fields in existing '
                  'data records or restrict data import to new records and '
                  'fill in missig information in existing records.'
                  '  By uploading a file no data will be deleted even if a '
                  'field in the uploaded table is empty.  Use the admin '
                  'interface to delete records or blank a field.',
    )
    erase_on_blank = forms.BooleanField(
        widget=forms.RadioSelect(choices=((True, 'erase'), (False, 'keep values'))),
        initial=False,
        required=False,
        label='blank value behavior',
        help_text='The default behavior when encountering blank/empty fields '
                  'in the input data is to ingnore these and keep the current'
                  ' value in the database.  This option allows to instead '
                  'erase the current value.',
    )


def get_field_search_form(field):
    """
    Factory to build field search forms
    """
    name = QUERY_FILTER + '-' + field + '__regex'
    field = forms.CharField(
        label=field.capitalize(),
        strip=True,
    )
    return type('FieldSearchForm', (forms.Form, ), {name: field})


def export_form_factory(view):
    """
    Return the export format form class for a given view

    :param TableView view: View whose table will be exported
    """
    query_dict = view.to_query_dict(fields=view.fields, keep_other=True)
    fields = query_dict.getlist(QUERY_FIELD)

    initial_fields = fields
    # prefer name over id
    if 'name' in view.fields and 'name' not in initial_fields:
        initial_fields.append('name')
    if 'name' in initial_fields:
        try:
            initial_fields.pop(initial_fields.index('id'))
        except ValueError:
            pass

    verbose_names = {
        i.name: i.verbose_name
        for i in view.model.get_fields().fields
    }

    choices = ((i, verbose_names.get(i, i)) for i in fields)
    opts = {}

    opts[QUERY_FIELD] = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=choices,
        initial=initial_fields,
        label='fields to be exported',
    )

    opts[QUERY_FORMAT] = forms.ChoiceField(
        widget=forms.RadioSelect,
        choices=[(i[0], i[2].description) for i in view.FORMATS],
        initial=view.DEFAULT_FORMAT,
        label='file format',
    )

    # Hidden fields to keep track of complete state, needed since the GET
    # action get a new query string attached, made entirely up from the form's
    # input elements
    for k, v in query_dict.lists():
        if k == QUERY_FIELD:
            continue
        opts[k] = forms.CharField(
            widget=forms.MultipleHiddenInput(),
            initial=v
        )

    return type('ExportForm', (forms.Form, ), opts)
