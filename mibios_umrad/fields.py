from pathlib import Path

from django.core.exceptions import ValidationError
from django.db.models import Field, CharField


def path_exists_validator(value):
    if not value.exists():
        raise ValidationError(f'path does not exist: {value}')


class PrefixValidator:
    prefix = ''

    def __init__(self, prefix):
        if prefix is not None:
            self.prefix = prefix

    def __call__(self, value):
        return value.startswith(self.prefix)


class AccessionField(CharField):
    description = 'dataset-specific accession'
    prefix = None
    DEFAULT_LENGTH = 32

    def __init__(self, verbose_name=None, name=None, prefix=None, **kwargs):
        # We have a default max length and also by default unique=True and no
        # prefix, name.  Verbose name will be generated from the model.
        # TODO: what about generating the name, too?
        kwargs.setdefault('max_length', self.DEFAULT_LENGTH)
        kwargs.setdefault('unique', True)
        super().__init__(verbose_name, name, **kwargs)

        if prefix is not None:
            self.prefix = prefix
            self.validators.append(PrefixValidator(prefix))

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if kwargs.get("max_length") == self.DEFAULT_LENGTH:
            del kwargs['max_length']
        if self.unique:
            del kwargs['unique']
        else:
            kwargs['unique'] = False
        if self.prefix is not None:
            kwargs['prefix'] = self.prefix
        return name, path, args, kwargs

    def contribute_to_class(self, cls, name, **kwargs):
        set_verbose_name = self.verbose_name is None
        super().contribute_to_class(cls, name, **kwargs)
        if set_verbose_name:
            # verbose_name should now have been set via
            # set_attributes_from_name()
            # FIXME: line below causes this default to be used in a model
            # derived from an abstract model that declares an AccessionField:
            # ??? self.verbose_name = cls._meta.verbose_name + ' id'
            pass

    def from_db_value(self, value, expression, connection):
        if self.prefix is not None and value is not None:
            return self.prefix + value
        return value

    def get_prep_value(self, value):
        if self.prefix is not None:
            if value.startswith(self.prefix):
                value = value[len(self.prefix):]
        return value


class PathField(Field):
    description = 'a file-system-path-like field'
    default_base = './q'  # str

    def __init__(self, *args, max_length=1024, base=None, exists=True,
                 null=True, default=None, **kwargs):
        super().__init__(*args, null=null, default=default, **kwargs)
        self.max_length = max_length
        if base is None:
            self.base = Path(self.default_base)
        else:
            self.base = Path(self.default_base) / base

        if exists:
            self.default_validators = [path_exists_validator]

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['base'] = str(self.base.relative_to(self.default_base))
        return name, path, args, kwargs

    def get_internal_type(self):
        return 'CharField'

    def from_db_value(self, value, expression, connection):
        if value is None:
            return None

        return self.base / value

    def to_python(self, value):
        if isinstance(value, Path):
            return value
        if value is None:
            return None
        # TODO: verify that this can never fail
        return Path(value)

    def get_prep_value(self, value):
        return str(value.relative_to(self.base))

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))
