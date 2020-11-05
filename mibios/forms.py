from collections import OrderedDict

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
        widget=forms.RadioSelect(
            choices=((True, 'overwrite'), (False, 'append-only')),
            attrs={'class': None},
        ),
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
        widget=forms.RadioSelect(
            choices=((True, 'erase'), (False, 'keep values')),
            attrs={'class': None},
        ),
        initial=False,
        required=False,
        label='blank value behavior',
        help_text='The default behavior when encountering blank/empty fields '
                  'in the input data is to ingnore these and keep the current'
                  ' value in the database.  This option allows to instead '
                  'erase the current value.',
    )
    allow_new_records = forms.BooleanField(
        required=False,
        initial=True,
        label='Allow new records',
        help_text='De-select this to prevent new records from being created '
                  'while still allowing modifications to existing records.',
    )


def get_field_search_form(*fields):
    """
    Factory to build field search forms

    :param fields: list of field names
    """
    opts = {}
    for i in fields:
        name = QUERY_FILTER + '-' + i + '__regex'
        field = forms.CharField(
            label=i.capitalize().replace('__', ' '),
            strip=True,
            required=False,
        )
        opts[name] = field
    return type('FieldSearchForm', (forms.Form, ), opts)


class ExportFormatForm(forms.Form):
    """
    Abstract base class for all export forms

    This class only implements the format choice field.
    """
    format_choices = ()
    """The available formats, need to be provided by implementing class"""

    default_format = None
    """Default choice of format to be provided by implementing class"""

    format = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': None}),
        # choices/initial set by constructor
        label='file format',
    )

    @classmethod
    def factory(cls, view, name=None, base=None, opts=OrderedDict()):
        """
        Return a form class to work with given ExportBaseMixin derived view.

        This method can be called as super().factory() from a deriving class
        and then will just add the parent attributes.

        Hidden fields to keep track of complete state, needed since the GET
        action get a new query string attached, made entirely up from the
        form's input elements.
        """
        query_dict = view.to_query_dict(fields=view.fields, keep=True)
        opts['format_choices'] = [
            (i[0], i[2].description)
            for i in view.FORMATS
        ]
        opts['default_format'] = view.DEFAULT_FORMAT
        # add hidden fields:
        for k, v in query_dict.lists():
            if k in [QUERY_FIELD, QUERY_FORMAT]:
                # should be provided by dedicated field
                continue
            opts[k] = forms.CharField(
                widget=forms.MultipleHiddenInput(),
                initial=v
            )

        name = name or 'Auto' + cls.__name__
        base = base or (cls, )
        return type(name, base, opts)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['format'].choices = self.format_choices
        self.fields['format'].initial = self.default_format

    def add_prefix(self, field_name):
        """
        API abuse to correctly set the HTML input attribute
        """
        if field_name == 'format':
            field_name = QUERY_FORMAT
        return super().add_prefix(field_name)


class ExportForm(ExportFormatForm):
    """
    Abstract class for all table export forms

    Inheriting classes should be defined by a factory that supplies the missing
    choice and initial attributes.
    """
    field_choices = ()
    initial_fields = ()

    exported_fields = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple(attrs={'class': None}),
        # choices/initial set by constructor
        label='fields to be exported',
    )

    @classmethod
    def factory(cls, view, name=None, base=None, opts=OrderedDict()):
        """
        Return a form class from the given ExportFormView

        """
        query_dict = view.to_query_dict(fields=view.fields, keep=True)
        fields = query_dict.getlist(QUERY_FIELD)

        initial_fields = list(fields)
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
            for i in view.model.get_fields(with_hidden=True).fields
        }

        field_choices = [(i, verbose_names.get(i, i)) for i in fields]
        opts['field_choices'] = field_choices
        opts['initial_fields'] = initial_fields

        name = name or 'Auto' + cls.__name__
        base = base or (cls, )
        return super().factory(view, name, base, opts)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['exported_fields'].choices = self.field_choices
        self.fields['exported_fields'].initial = self.initial_fields
        # order field choices before format choices:
        self.order_fields(('exported_fields', 'format'))

    def add_prefix(self, field_name):
        """
        API abuse to correctly set the HTML input attribute
        """
        if field_name == 'exported_fields':
            field_name = QUERY_FIELD
        return super().add_prefix(field_name)
