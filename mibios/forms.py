from django import forms


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
