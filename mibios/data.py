"""
Module for data abstraction
"""
from urllib.parse import urlparse

from django import forms
from django.http.request import QueryDict
from django.utils.text import slugify
from django.urls import resolve, reverse

from . import (get_registry, QUERY_FILTER, QUERY_EXCLUDE, QUERY_NEGATE,
               QUERY_SHOW, QUERY_COUNT, QUERY_Q)
from .dataset import Dataset
from .models import Model, Q
from .tables import NONE_LOOKUP
from .utils import getLogger


log = getLogger(__name__)


NO_CURATION_PREFIX = 'not-curated-'


class DataConfig:
    """
    A representation of a selection of data
    """
    def __init__(self, arg, show_hidden=False):
        # must set show_hidden in init to get correct set of defaults fields
        # TODO: dynamically add/remove hidden fields
        self.show_hidden = show_hidden

        self._set_name(arg)

        # settings unrelated to name/model/dataset:
        self.is_curated = True
        self.extras = {}

    @classmethod
    def from_url(cls, url):
        """
        Return instance corresponding to given URL

        This is intended for testing and debugging.  Missing curation status
        handling.
        """
        url = urlparse(url)
        data_name = resolve(url.path).kwargs['data_name']
        conf = cls(data_name)
        conf.set(QueryDict(url.query))
        return conf

    def _copy(self):
        """
        Make and return a deepish copy of instance
        """
        obj = type(self)(self.name)
        for k, v in vars(self).items():
            if k == 'excludes':
                # is list of dicts
                v = [i.copy() for i in v]
            elif isinstance(v, (list, dict)):
                v = v.copy()  # shallow copies
            setattr(obj, k, v)
        return obj

    def _set_name(self, arg):
        """
        Set up config with given name

        This will clear all model-specific settings. Inheriting classes should
        override this method, call via super() and reset model-specific
        settings as needed.
        """
        self.verbose_name = None
        self.fields = []
        self.fields_verbose = []
        self.avg_by = None
        self._manager = None
        self.clear_selection()
        if isinstance(arg, str):
            name = arg
            reg = get_registry()
            # name of model or dataset
            try:
                dataset = reg.datasets[name]
            except KeyError:
                try:
                    model = reg.models[name]
                except KeyError:
                    raise LookupError(f'no such dataset or model: {name}')
                else:
                    self._setup_model(model)
            else:
                self._setup_dataset(dataset)
        elif isinstance(arg, Dataset):
            self._setup_dataset(arg)
        elif isinstance(arg, Model):
            self._setup_model(arg)
        else:
            raise TypeError('arg must be str or Dataset or Model')

    def _setup_dataset(self, dataset):
        self.name = dataset.name
        self.verbose_name = dataset.name
        self.model = dataset.model
        if dataset.manager:
            self._manager = getattr(self.model, dataset.manager)
        self.filter = dataset.filter
        self.excludes = dataset.excludes
        for i in dataset.fields:
            try:
                name, verbose = i
            except ValueError:
                # assume one-tuple
                name = i[0]
                verbose = i[0]
            except TypeError:
                # assume i is str
                name = i
                verbose = i

            self.fields.append(name)
            self.fields_verbose.append(verbose)

    def _setup_model(self, model):
        self.model = model
        self.name = model._meta.model_name
        self.verbose_name = model._meta.verbose_name
        self.set_fields()

    def set_fields(self, selected_fields=[]):
        """
        Sets fields

        :param list selected_fields:
            List of names to which the fields shall be restricted.  Unknown
            field names are ignored.

        Sets the fields/columns to show to.  By default, all "simple" fields of
        the model are shown. The default case also ensures that a name/id field
        is set.
        """
        self.fields = []
        self.fields_verbose = []
        has_name_field = False

        if selected_fields:
            choices = []
            for i in self.model.get_related_accessors():
                try:
                    _f = self.model.get_field(i)
                except LookupError:
                    continue  # TODO: handle name natural?

                try:
                    _v = _f.verbose_name
                except AttributeError:
                    # _f is OneToOneRel (FIXME)
                    _v = _f.name

                choices.append((i, _v))
            del _f, _v
        else:
            fields = self.model.get_fields(with_hidden=self.show_hidden)
            choices = zip(fields.names, fields.verbose)
            del fields

        for name, verbose_name in choices:
            if selected_fields and name not in selected_fields:
                continue
            if name == 'name':
                has_name_field = True
            self.fields.append(name)
            if name == verbose_name:
                # None: will be capitalized by django-tables2
                self.fields_verbose.append(None)
            else:
                # e.g. when letter case is important, like for 'pH'
                self.fields_verbose.append(verbose_name)
        del name, verbose_name

        if hasattr(self.model, 'name'):
            try:
                id_pos = self.fields.index('id')
            except ValueError:
                id_pos = 0
            else:
                # hide internal IDs if we have some "name"
                # (but not natural, TODO: review this design decision)
                self.fields.pop(id_pos)
                self.fields_verbose.pop(id_pos)
            if not has_name_field:
                # replace id column with name property
                self.fields.insert(id_pos, 'name')
                self.fields_verbose.insert(id_pos, None)
            del id_pos

    def get_queryset(self):
        excludes = [~Q(**i, model=self.model) for i in self.excludes]
        q = Q(*self.q, *excludes, **self.filter, model=self.model)

        if self.negate:
            q = ~q

        related_fields = []
        for i in self.fields:
            try:
                f = self.model.get_field(i)
            except LookupError:
                continue
            if f.is_relation and not f.many_to_many:
                related_fields.append(i)
                for j in f.related_model.resolve_natural_lookups('natural'):
                    if '__' in j:
                        # need following relation for natural key
                        related_fields.append(
                            i + '__' + j.rpartition('__')[0]
                        )
            del f

        log.debug(f'Dataconfig: get_queryset: Q: {q}, rel:{related_fields}')

        if self._manager is None:
            if self.is_curated:
                qs = self.model.curated.all()
            else:
                qs = self.model.objects.all()
        else:
            qs = self._manager.all()

        qs = qs.select_related(*related_fields).filter(q)

        if self.avg_by is not None:
            qs = qs.average(*self.avg_by)

        return qs

    def clear_selection(self):
        self.q = []
        self.excludes = []
        self.filter = {}
        self.negate = False

    def set_name(self, name):
        """
        Switch config over to different model/dataset

        Will wipe all settings related to the model/dataset

        Returns a new instance.
        """
        obj = self._copy()
        obj._set_name(name)
        return obj

    def set(self, qdict):
        """
        Update from a GET querydict

        :param QueryDict qdict: A QueryDict usually obtained via request.GET

        Existing filter and excludes are preserved, new filters are added.  Any
        unprocessed qdict key/list pairs are added (update) to the extras
        dictionary to give child classed a chance of processing them.
        Accordingly, overriding methods should first call super().set(qdict)
        before proceeding.
        """
        qlist = []
        filter = {}
        excludes = {}
        negate = False
        extras = {}

        for qkey, val_list in qdict.lists():
            # TODO: error handling
            if qkey.startswith(QUERY_FILTER + '-'):
                _, _, filter_key = qkey.partition('-')
                val = val_list[-1]
                if val == NONE_LOOKUP:
                    val = None
                filter[filter_key] = val

            elif qkey.startswith(QUERY_EXCLUDE + '-'):
                _, idx, exclude_key = qkey.split('-')
                val = val_list[-1]
                if val == NONE_LOOKUP:
                    val = None
                if idx not in excludes:
                    excludes[idx] = {}
                excludes[idx][exclude_key] = val

            elif qkey == QUERY_NEGATE:
                val = val_list[-1]
                # test for presence/absence, other values are invalid
                if val == '':
                    negate = True

            elif qkey == QUERY_Q:
                for i in val_list:
                    qlist.append(Q.deserialize(i))

            else:
                extras[qkey] = val_list

        # convert excludes into list, forget the index
        excludes = [i for i in excludes.values()]
        log.debug('DECODED QUERYSTRING:', filter, excludes, negate, extras)

        self.q = qlist
        self.excludes += excludes
        self.filter.update(**filter)
        self.negate = negate
        self.extras = extras

    def as_query_dict(self):
        """
        Convert instance into corresponding QueryDict
        """
        qdict = QueryDict(mutable=True)

        have_filter_or_excl = False
        for k, v in self.filter.items():
            k = slugify((QUERY_FILTER, k))
            if v is None:
                v = NONE_LOOKUP
            qdict[k] = v
            have_filter_or_excl = True

        for i, excl in enumerate(self.excludes):
            for k, v in excl.items():
                k = slugify((QUERY_EXCLUDE, i, k))
                if v is None:
                    v = NONE_LOOKUP
                qdict[k] = v
                have_filter_or_excl = True

        if self.q:
            qdict.setlist(QUERY_Q, [i.serialize() for i in self.q])

        if have_filter_or_excl and self.negate:
            # only add negate if we have some filtering
            qdict[QUERY_NEGATE] = ''

        qdict = self._populate_query_dict(qdict)

        for k, v in self.extras.items():
            if isinstance(v, str):
                # avoid setlist on str
                qdict[k] = v
            else:
                try:
                    qdict.setlist(k, v)
                except TypeError:
                    # v is not iterable
                    qdict[k] = v

        qdict._mutable = False
        return qdict

    def url_path(self):
        """
        Return url path
        """
        kwargs = {}

        if self.avg_by:
            kwargs['avg_by'] = self.avg_by
            url_name = 'average'
        else:
            url_name = 'table'

        if self.is_curated:
            kwargs['data_name'] = self.name
        else:
            kwargs['data_name'] = NO_CURATION_PREFIX + self.name

        return reverse(url_name, kwargs=kwargs)

    def url_query(self):
        """
        URL-encode url query part

        A convenience method, the output is intended to be rendered inside a <a
        href="?{{ }}"> element.
        """
        return self.as_query_dict().urlencode(safe=',')

    def url(self):
        """
        Full URL for this DataConfig
        """
        querystr = self.url_query()
        return self.url_path() + ('?' + querystr if querystr else '')

    def as_hidden_input(self, skip=[]):
        """
        Make fields to be used as hidden form input
        """
        opts = {}
        for k, v in self.as_query_dict().lists():
            if k in skip:
                # is provided by caller
                continue

            opts[k] = forms.CharField(
                widget=forms.MultipleHiddenInput(),
                initial=v
            )
        return opts

    def _populate_query_dict(self, qdict):
        """
        Child classes should override this method to populate query dict
        """
        return qdict

    def add_filter(self, **kwargs):
        """
        Add more constraints to filter

        Returns a new instance.
        """
        obj = self._copy()
        obj.filter.update(**kwargs)
        return obj

    def add_exclude(self, **kwargs):
        """
        Add an exclude

        Returns a new instance.
        """
        obj = self._copy()
        obj.excludes.append(kwargs)
        return obj

    def remove_filter(self, **items):
        """
        Return a copy config with given filters removed
        """
        obj = self._copy()
        for k, v in items.items():
            if k in obj.filter and obj.filter[k] == v:
                del obj.filter[k]
        return obj

    def remove_excludes(self, *items):
        """
        Return a copy config with given excludes removed
        """
        obj = self._copy()
        obj.excludes = [j for j in obj.excludes if j not in items]
        return obj

    def put(self, **kwargs):
        """
        Return copy of instance with given changes applied
        """
        obj = self._copy()
        for k, v in kwargs.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
            else:
                obj.extras[k] = v
        return obj

    def inverse(self):
        """
        Return inversed/negated clone of instance

        A convenience method, so one can put view.conf.inverse.url straight
        into a template.
        """
        return self.put(negate=not self.negate)


class TableConfig(DataConfig):
    """
    Represents data selection and options for a table
    """
    @property
    def show(self):
        """ A selection of fields """
        return self._show

    @show.setter
    def show(self, field_list):
        """ Setting show will also update the fields """
        self._show = field_list
        self.set_fields(field_list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_counts = False

    def _set_name(self, *args, **kwargs):
        super()._set_name(*args, **kwargs)
        self.show = []
        self.sort_by_field = None  # TODO: implement handling sort

    def get_queryset(self):
        qs = super().get_queryset()

        if self.with_counts:
            qs = qs.annotate_rev_rel_counts()
        return qs

    def set(self, *args):
        """
        Update from a GET querydict

        :param QueryDict qdict: A QueryDict usually obtained via request.GET
        """
        super().set(*args)
        self.with_counts = False
        show = []
        extras = {}

        for qkey, val_list in self.extras.items():
            if qkey == QUERY_SHOW:
                show += val_list
            elif qkey == QUERY_COUNT:
                if val_list == ['']:
                    self.with_counts = True
                else:
                    # invalid, ignore
                    pass

            else:
                extras[qkey] = val_list

        self.show = show
        self.extras = extras

    def _populate_query_dict(self, qdict):
        if self.show:
            qdict.setlist(QUERY_SHOW, self.show)

        if self.with_counts:
            qdict[QUERY_COUNT] = ''

        return qdict
