"""
    rvmath
    ~~~~~~

    Evaluate expression involving random variables.

    :copyright: 2021 by rvmath Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations


def __dir__():  # pragma: no cover
    from scipy.stats import _continuous_distns

    return _continuous_distns._distn_names + ["wrap"]


def __getattr__(name):
    from scipy import stats

    from .base import builder, wrap

    if name == "wrap":
        return wrap

    distro_cls = getattr(stats, name, None)

    if distro_cls is None:
        raise AttributeError(f"module 'rvmath' has no attribute '{name}'")

    return builder(distro_cls)
