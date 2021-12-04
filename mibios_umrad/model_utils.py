"""
Model-related utilities

Separate module to avoid circular imports issues
"""
from django.db import models


# standard data field options
opt = dict(blank=True, null=True, default=None)  # non-char optional
ch_opt = dict(blank=True, default='')  # optional char
uniq_opt = dict(unique=True, **opt)  # unique and optional (char/non-char)
# standard foreign key options
fk_req = dict(on_delete=models.CASCADE)  # required FK
fk_opt = dict(on_delete=models.SET_NULL, **opt)  # optional FK
