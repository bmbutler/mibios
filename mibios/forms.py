from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField()

    def clean_data(self, *args, **kwargs):
        ret = super().clean_data(*args, **kwargs)
        print('CLEAN DATA', ret)
        return ret
