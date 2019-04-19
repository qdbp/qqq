from __future__ import annotations

import os.path as osp
import pickle
import sys
import typing as ty
from collections import defaultdict, namedtuple
from collections.abc import Mapping, Sequence
from functools import wraps
from inspect import Parameter, signature
from time import sleep, time
from typing import Any, Collection, Dict, List, Optional, Sized, TypeVar

PrioTup = namedtuple("PrioTup", ("prio", "value"))

T = TypeVar('T')


def die(msg, code=-1, debug_locals=None):
    print(msg, file=sys.stderr)
    if debug_locals:
        from traceback import print_stack
        from pprint import pprint
        print_stack()
        pprint(debug_locals)
    sys.exit(code)


def ensure_type(obj, t, *args, **kwargs):
    """
    Validate that `obj` is an object of type `t`.

    If `obj` is None, the constructor `t` is called with *args, *kwargs.

    If `obj` is not None and not an instance of type `t`, ValueError is raised.
    """

    if obj is None:
        return t(*args, **kwargs)
    elif not isinstance(obj, t):
        raise ValueError(f"need a {t.__name__} object, got {obj}")
    else:
        return obj


def as_list(obj: Optional[ty.Union[T, List[T], ty.Tuple[T], ty.Set[T]]]
            ) -> List[T]:
    """
    Coerces the input into a list.

    None is turned into an empty list.
    Tuples and Sets are coerced into a list.
    All other objects are wrapped in a singleton list.
    """

    if obj is None:
        return []
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, list):
        return obj
    else:
        return [obj]


def flatten(xs: List[List[T]]) -> List[T]:
    """
    Flattens a list of lists. Does not recurse.
    """
    return sum(xs, [])


def sift_kwargs(f):
    """
    Lets the wrapped function silently ignore invalid kwargs.
    """

    @wraps(f)
    def _f(*args, **kwargs):
        return f(*args, **kwsift(kwargs, f))

    return _f


def kwsift(kw, f):
    """
    Sifts a keyword argument dictionary with respect to a function.

    Returns a dictionary with those entries that the given function
    accepts as keyword arguments.

    If the function is found to accept a variadic keyword dictionary
    (**kwargs), the first argument is returned unchanged, since any keyword
    argument is therefore legal.
    """

    sig = signature(f)
    kw_kinds = {Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    out = {}
    # go backward to catch **kwargs on the first pass
    for name, p in list(sig.parameters.items())[::-1]:
        if p.kind == p.VAR_KEYWORD:
            return kw
        elif p.kind in kw_kinds and name in kw.keys():
            out[name] = kw[name]

    return out


def check_all_same_length(
        *args: Sized, allow_none: bool = False, msg: str = None
) -> int:
    """
    Checks that all arguments are the same length.  Raises ValueError if
    arguments' lengths differ.

    Returns arguments' shared length.
    """
    if not args:
        return 0

    s = {len(arg) for arg in args if not allow_none or arg is not None}

    if len(s) != 1:
        raise ValueError(
            f"arguments have different lengths! {s}\n" + (msg or "")
        )

    return s.pop()


def alleq(xs: Collection) -> bool:
    """
    Returns true iff all elements of the Collection `xs` are equal
    """
    if len(xs) == 0:
        return True
    else:
        it = iter(xs)
        x0 = next(it)
        return all(x == x0 for x in it)


def pickled(fn, func, *args, **kwargs):
    if osp.isfile(fn):
        return load_pickle(fn)
    out = func(*args, **kwargs)
    save_pickle(out, fn)
    return out


def save_pickle(obj, name):
    with open(f"{name}.p", "wb") as f:
        pickle.dump(obj, f)


def load_pickle(name):
    with open(f"{name}.p", "rb") as f:
        return pickle.load(f)
