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
        help_text='whether to allow overwriting existing data (when your data '
                  'differs from what\'s in the database and the field in the '
                  'database is not empty) or just fill in missing information.'
                  '  By uploading a file no data will be deleted even if a '
                  'field in the uploaded table is empty.',
    )
