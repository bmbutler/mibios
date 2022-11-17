from django.core.validators import URLValidator
from django.db.models import URLField


class OptionalHTTPSURLValidator(URLValidator):
    """ Validates https:// URLs while allowing blanks """
    schemes = ['https']

    def __call__(self, value):
        if value == '':
            # Allow passing a blank field with a valid example URL
            value = 'https://example.net'
        super().__call__(value)


class OptionalURLField(URLField):
    """ Field for https:// URLs but may remain blank """
    default_validators = [OptionalHTTPSURLValidator]

    def __init__(self, **kwargs):
        kwargs.setdefault('blank', True)
        super().__init__(**kwargs)
