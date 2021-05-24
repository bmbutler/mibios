from collections import OrderedDict

from django import forms

from . import QUERY_FILTER, QUERY_FORMAT
from .utils import getLogger


log = getLogger(__name__)


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


def get_field_search_form(table_conf, *fields):
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

    opts.update(**table_conf.as_hidden_input(skip=['name']))

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
        base = base or (cls, )
        query_dict = view.conf.as_query_dict()
        opts['format_choices'] = [
            (i[0], i[2].description)
            for i in view.FORMATS
        ]
        opts['default_format'] = view.DEFAULT_FORMAT

        # add hidden fields:
        for k, v in query_dict.lists():
            if k in [QUERY_FORMAT]:
                # should be provided by dedicated field
                continue

            if k in cls.base_fields.keys():
                # is already a field declared by class
                continue

            opts[k] = forms.CharField(
                widget=forms.MultipleHiddenInput(),
                initial=v
            )

        name = name or 'Auto' + cls.__name__
        return type(name, base, opts)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['format'].choices = self.format_choices
        self.fields['format'].initial = self.default_format

    def order_fields(self, field_order):
        """
        Order format field last unless specified otherwise

        Without this format would always be first in the form.
        """
        super().order_fields(field_order)
        if field_order and 'format' in field_order:
            # order was given explicitly
            return

        if 'format' in self.fields:
            self.fields['format'] = self.fields.pop('format')

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

    show = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple(attrs={'class': None}),
        # choices/initial set by constructor
        label='fields to be exported',
    )

    @classmethod
    def factory(cls, view, name=None, base=None, opts=OrderedDict()):
        """
        Return a form class from the given ExportFormView

        """
        fields = view.conf.fields

        if view.conf.show:
            initial_fields = view.conf.show
        else:
            initial_fields = view.conf.fields
        # prefer name over id
        if 'name' in view.conf.fields and 'name' not in initial_fields:
            initial_fields.append('name')
        if 'name' in initial_fields:
            try:
                initial_fields.pop(initial_fields.index('id'))
            except ValueError:
                pass

        # TODO: allow export of hidden fields, requies Dataconfig to be
        # switchable between show and hiden hidden fields
        # This is a regression of the Dataconfig transition
        field_choices = [
            (i, i if j is None else j)
            for i, j
            in zip(fields, view.conf.fields_verbose)
        ]
        opts['field_choices'] = field_choices
        opts['initial_fields'] = initial_fields

        name = name or 'Auto' + cls.__name__
        base = base or (cls, )
        return super().factory(view, name, base, opts)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['show'].choices = self.field_choices
        self.fields['show'].initial = self.initial_fields
        # order field choices before format choices:
        self.order_fields(('show', 'format'))


class ShowHideForm(forms.Form):
    # field name should equal value of mibios.QUERY_SHOW
    show = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple(attrs={'class': None}),
        # choices set by constructor
        label='show these fields/columns',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['show'].choices = self.choices

    @classmethod
    def factory(cls, data_conf):
        """
        Build form listing all forward related fields
        """
        model = data_conf.model
        choices = []
        for i in model.get_related_accessors():
            if i.endswith('__natural'):
                # DatasetMixin expects the foreign key relation
                i = i[:-len('__natural')]
            path = i.split('__')
            try:
                path[-1] = model.get_field(i).verbose_name
            except LookupError:
                # natural or name property
                if i in ['natural', 'name']:
                    try:
                        model.get_field('name')
                    except LookupError:
                        if hasattr(model, 'name'):
                            i = 'name'

                        path = [model._meta.model_name]
                    else:
                        # name field supercedes natural
                        continue
                else:
                    raise
            except Exception:
                # e.g. caused by OneToOneRel not having a verbose_name
                pass
            verbose_path = ' > '.join(path)
            choices.append((i, verbose_path))

        opts = dict(
            choices=tuple(choices),
        )
        opts.update(**data_conf.as_hidden_input(skip=['show']))
        return type('AutoShowHideForm', (cls,), opts)
