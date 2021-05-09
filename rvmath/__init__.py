"""
    rvmath
    ~~~~~~

    Evaluate expression involving random variables.

    :copyright: 2021 by rvmath Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations


def __dir__():
    from scipy.stats import _continuous_distns

    return _continuous_distns._distn_names + ["One"]


def __getattr__(name):
    from .base import DependentRandomVariable, One, RandomVariable

    if name == "One":
        return One

    import itertools as it

    from scipy import stats

    parent = getattr(stats, name, None)

    if parent is None:
        raise AttributeError(f"module 'rvmath' has no attribute '{name}'")

    # Check if this is a continuous distribution

    def builder(*args, **kwargs):
        rvid = kwargs.pop("rvid", None)

        if any(isinstance(a, RandomVariable) for a in it.chain(args, kwargs.values())):
            if rvid is None:
                return DependentRandomVariable(parent, args=args, kwds=kwargs)
            else:
                return DependentRandomVariable(
                    parent, rvid=rvid, args=args, kwds=kwargs
                )

        distro = parent(*args, **kwargs)

        if rvid is None:
            return RandomVariable(distro)
        else:
            return RandomVariable(distro, rvid=rvid)

    return builder
