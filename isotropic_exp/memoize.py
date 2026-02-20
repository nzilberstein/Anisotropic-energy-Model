""" Small utility functions for memoization. """

import hashlib
from pathlib import Path
from typing import *

import torch

from printing import kwargs_to_str

def memory_memoize(cache: dict, key, closure, recompute=False):
    """ Memoizes the result of closure in the provided cache.
    :param cache: dictionary, name -> result
    :param key: key associated to given closure
    :param closure: callable with no parameters, returns result
    :param recompute: whether to recompute the result even if it is in cache
    :return: result of closure
    """
    if recompute or key not in cache:
        cache[key] = closure()

    return cache[key]


def disk_memoize(path: Path | str, closure, recompute=False):
    """ Memoizes the result of closure on disk.
    :param path: path to the file used for saving the result
    :param closure: callable with no parameters, returns result
    :param recompute: whether to recompute the result even if it is on disk
    :return: result of closure
    """
    if not isinstance(path, Path):
        path = Path(path)

    if (not recompute) and path.exists():
        result = torch.load(str(path), weights_only=False)
    else:
        result = closure()
        torch.save(result, str(path))

    return result


def short_hash(string, length=8):
    """ Return a short hash of a string, for shortening file names used for memoization. """
    h = hashlib.md5()
    h.update(string.encode())
    return h.hexdigest()[-length:]


def short_name(name, **kwargs):
    """ Returns a short filename of the form `name_HASH.pt` by hashing provided kwargs (order-sensitive). """
    return f"{name}_{short_hash(kwargs_to_str(kwargs))}.pt"
