"""
    rvmath.base
    ~~~~~~~~~~~

    :copyright: 2021 by rvmath Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numbers
import operator
import secrets
import typing as ty
from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate
from scipy import stats

RVID_NBYTES = 16

_OP_STR = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.pow: "**",
    operator.pos: "+",
    operator.neg: "-",
}


def ecdf(x):
    """Empirical from cumulative distribution function.

    Parameters
    ----------
    x : array-like
        data
    Returns
    -------
    np.ndarray, np.ndarray
        value, ecdf
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def eval_value(value, realization):
    """Helper to dispatch the evaluation of (maybe) RVMixin values

    See RVMixin.eval for help on `realization`.
    """
    if isinstance(value, RVMixin):
        return value.eval(realization)
    return value


def any_none(els):
    """Return True if any of the elements is None."""
    return any(el is None for el in els)


def combine_size(distro_size, size):
    """Combine distribution and user size according to certain rules.

    Parameters
    ----------
    distro_size : None, int or tuple of int or None
        Size assigned to the distribution.
    size : int or tuple of int
        Size provided to the `rvs`.

    Returns
    -------
    int or tuple of int
    """

    if size is None:
        raise ValueError("'size' cannot be None.")
    elif isinstance(size, tuple):
        if any_none(size):
            raise ValueError("'size' cannot contain None.")

    if distro_size is None:
        return size

    elif isinstance(distro_size, tuple) and isinstance(size, tuple):
        if any_none(distro_size):
            raise ValueError(
                "A distribution 'distro_size' cannot contain None "
                "when the 'rvs' distro_size is a tuple."
            )
        return distro_size

    elif isinstance(distro_size, tuple) and isinstance(size, int):
        return tuple(el or size for el in distro_size)

    return distro_size


class RVMixin:
    """Mixin for classes that are or can contain random variables."""

    def random_vars(self) -> ty.Tuple[str, stats.rv_continuous]:
        """Yields all random variables and their distributions within this expression.

        Yields
        ------
        str, stats.rv_continuous
            variable name, distribution
        """

        # This weird construction is a way to create
        # an empty generator.
        return
        yield  # pragma: no cover

    def eval(self, realization):
        """Evaluate this expression given a realization of its random variables.

        Parameters
        ----------
        realization : Dict[str, np.ndarray or Number]
            Dictionary mapping random variable id to a realization.

        Returns
        -------
        np.ndarray or Number
        """
        raise NotImplementedError

    def draw(self, size=1, random_state=None):
        """Draw values for the random variables within this expression."""
        return {
            rvid: distro.rvs(combine_size(sz, size), random_state)
            for rvid, (distro, sz) in self.random_vars()
        }

    def rvs(self, size=1, random_state=None):
        """
        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : None, int, RandomState, Generator, optional
            If seed is None the RandomState singleton is used. If seed is an int,
            a new RandomState instance is used, seeded with seed. If seed is already
            a RandomState or Generator instance, then that object is used. Default is None.

        Returns
        -------
        ndarray or number
            Random variates of given size.
        """
        return self.eval(self.draw(size, random_state))

    def to_distro(self, name, n=1_000_000, **kwargs):
        """Converts the current expression into a Random Variable Continuous distribution.
        (Scipy.stats.rv_continuous).

        This is done by estimating the CDF by drawing random samples and then building an interpolator.

        Parameters
        ----------
        name : str
            name of the distribution
        n : int, optional
            number of random samples to drawn from which the cdf
            is estimated (default: 1_000_000)
        kwargs:
            extra keyword arguments, passed directly to the
            distribution constructors

        """

        values = self.rvs(n)
        itp = scipy.interpolate.interp1d(
            *ecdf(values),
            copy=True,
            bounds_error=False,
            fill_value=(0, 1),
            assume_sorted=True,
        )

        class distro_gen(stats.rv_continuous):
            def _cdf(self, x):
                return itp(x)

        return distro_gen(name=name, **kwargs)()


class OperatorMixin:
    """Mixin used for to deal with math expression and function calls."""

    def __add__(self, other):
        return BinaryOp(operator.add, self, other)

    def __radd__(self, other):
        return BinaryOp(operator.add, other, self)

    def __sub__(self, other):
        return BinaryOp(operator.sub, self, other)

    def __rsub__(self, other):
        return BinaryOp(operator.sub, other, self)

    def __mul__(self, other):
        return BinaryOp(operator.mul, self, other)

    def __rmul__(self, other):
        return BinaryOp(operator.mul, other, self)

    def __truediv__(self, other):
        return BinaryOp(operator.truediv, self, other)

    def __rtruediv__(self, other):
        return BinaryOp(operator.truediv, other, self)

    def __pow__(self, power, modulo=None):
        return BinaryOp(operator.pow, self, power)

    def __rpow__(self, power, modulo=None):
        return BinaryOp(operator.pow, power, self)

    def __pos__(self):
        return UnaryOp(operator.pos, self)

    def __neg__(self):
        return UnaryOp(operator.neg, self)

    def __array_function__(self, func, types, args, kwargs):
        return Function(func, args, kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            return Function(ufunc, inputs, kwargs)
        else:
            return NotImplemented


Operand = ty.Union[numbers.Number, RVMixin, OperatorMixin]


@dataclass(frozen=True)
class ArgLessFunction(OperatorMixin, RVMixin):
    """An argument less function"""

    func: ty.Callable

    def eval(self, realization):
        return self.func()


@dataclass(frozen=True)
class WithArg(RVMixin):
    """Add arguments and keyword arguments handling to
    other dataclass
    """

    args: ty.Tuple[ty.Any] = field(default_factory=tuple)
    kwds: ty.Dict[str, ty.Any] = field(default_factory=dict)

    def random_vars(self):
        yield from super().random_vars()
        for arg in self.args:
            if isinstance(arg, RVMixin):
                yield from arg.random_vars()

        for k, v in self.kwds.items():
            if isinstance(v, RVMixin):
                yield from v.random_vars()

    def get_args_kwds(self, realization):
        args = tuple(eval_value(arg, realization) for arg in self.args)
        kwds = {k: eval_value(v, realization) for k, v in self.kwds.items()}
        return args, kwds


@dataclass(frozen=True)
class Function(WithArg, ArgLessFunction):
    """A function that can handles arguments and keyword arguments."""

    def eval(self, realization):
        args, kwds = self.get_args_kwds(realization)

        return self.func(*args, **kwds)


@dataclass(frozen=True)
class RandomVariable(OperatorMixin, RVMixin):
    """A random variable."""

    distro: stats.rv_continuous
    size: ty.Optional[numbers.Integral] = None
    rvid: str = field(default_factory=lambda: secrets.token_hex(nbytes=RVID_NBYTES))

    def random_vars(self):
        yield self.rvid, (self.distro, self.size)

    def eval(self, realization):
        return realization[self.rvid]

    def __str__(self):
        obj = self.distro
        s = tuple((str(a) for a in obj.args)) + tuple(
            (f"{k}= {v}" for k, v in obj.kwds)
        )
        return f"{obj.dist.name}({', '.join(s)})#{self.rvid}"


@dataclass(frozen=True)
class DependentRandomVariable(WithArg, RandomVariable):
    """A random variable that depends on other random variables
    (e.g. it's mean value is drawn from another ramdom variable).
    """

    def __str__(self):
        obj = self.distro
        s = tuple((str(a) for a in self.args)) + tuple(
            (f"{k}= {v}" for k, v in self.kwds)
        )
        return f"{obj.name}({', '.join(s)})#{self.rvid}"


@dataclass(frozen=True)
class UnaryOp(OperatorMixin, RVMixin):
    """An unary operator."""

    op: ty.Callable
    value: Operand

    def random_vars(self):
        if isinstance(self.value, RVMixin):
            yield from self.value.random_vars()

    def eval(self, realization):
        return self.op(eval_value(self.value, realization))

    def __str__(self):
        return _OP_STR[self.op] + str(self.value)


@dataclass(frozen=True)
class BinaryOp(OperatorMixin, RVMixin):
    """An binary operator."""

    op: ty.Callable
    value1: Operand
    value2: Operand

    def random_vars(self):
        if isinstance(self.value1, RVMixin):
            yield from self.value1.random_vars()

        if isinstance(self.value2, RVMixin):
            yield from self.value2.random_vars()

    def eval(self, realization):
        return self.op(
            eval_value(self.value1, realization),
            eval_value(self.value2, realization),
        )

    def __str__(self):
        return str(self.value1) + " " + _OP_STR[self.op] + " " + str(self.value2)


One = UnaryOp(operator.pos, 1)
