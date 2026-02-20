""" Printing, logging, and debugging utilities. """

from typing import *
import math


def tensor_summary_stats(x, format=".3f"):
    """ Returns summary statistics about x: shape, mean, std, min, max. """
    return f"shape {x.shape}, values in [{x.min():{format}}, {x.max():{format}}] and around {x.mean():{format}} +- {x.std():{format}}"


def shapes(x, names=None):
    """ Traverse x as nested lists/dictionaries of arrays/tensors and return the shapes of those.
    If names is provided, it should have the same structure as x and give the names of axes of tensors.
    """
    if isinstance(x, list):
        if names is None:
            return [shapes(v) for v in x]
        else:
            assert isinstance(names, list) and len(x) == len(names)
            return [shapes(x[i], names[i]) for i in range(len(x))]
    elif isinstance(x, dict):
        if names is None:
            return {k: shapes(v) for k, v in x.items()}
        else:
            assert isinstance(names, dict) and x.keys() == names.keys()
            return {k: shapes(x[k], names[k]) for k in x.keys()}
    elif x is None:
        assert names is None
        return None
    else:
        if names is None:
            return tuple(x.shape)
        else:
            assert len(names) == len(x.shape)
            return ", ".join(f"{names[i]}: {x.shape[i]}" for i in range(len(x.shape)))


def kwargs_to_str(kwargs):
    """ Returns a string of the form '(kw1=val1, kw2=val2)'. """
    if len(kwargs) == 0:
        return ""
    else:
        return "(" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"


def str_concat(substrings, sep=" ") -> str:
    """ Filter Nones, map to strings, and concatenate with a separator. """
    return sep.join(map(str, filter(None, substrings)))


def format_sig_digits(num, digits=1):
    """ Format a floating point number to some number of significant digits. """
    if num == 0:
        return '0'
    magnitude = math.floor(math.log10(abs(num)))
    rounded = round(num, digits - 1 -magnitude)
    return f'{rounded:f}'.rstrip('0').rstrip('.')
