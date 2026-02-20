""" Good old utility functions that can't really be grouped thematically. """

from typing import *


def ceil_div(a: int, b: int) -> int:
    """ Return ceil(a / b). """
    return a // b + (a % b > 0)


def to_tuple(x):
    """ Convert iterable to tuple, or make a tuple with one element. """
    if hasattr(x, "__iter__"):
        return tuple(x)
    else:
        return (x,)


def si_to_num(s: str, type=int) -> int:
    """ Convert a string potentially ending in a SI prefix to an integer or a float. """
    si_prefixes = {" ": 1e0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18, "Z": 1e21, "Y": 1e24}
    num, si = s[:-1], s[-1].upper()
    if si in si_prefixes:
        return type(num) * type(si_prefixes[si])
    else:
        return type(num)
