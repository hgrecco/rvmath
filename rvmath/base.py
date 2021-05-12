"""
    rvmath.base
    ~~~~~~~~~~~

    :copyright: 2021 by rvmath Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import itertools as it
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


def builder(distro_cls):
    """Creates a hungry wrapper function.

    Parameters
    ----------
    distro_cls : rv_continuous
        A SciPy distribution

    """

    # Check if this is a continuous distribution?

    def _inner(*args, **kwargs):
        rvid = kwargs.pop("rvid", None)
        size = kwargs.pop("size", None)

        if any(isinstance(a, RandomVariable) for a in it.chain(args, kwargs.values())):
            if rvid is None:
                return DependentRandomVariable(
                    distro_cls, size=size, args=args, kwds=kwargs
                )
            else:
                return DependentRandomVariable(
                    distro_cls, size=size, rvid=rvid, args=args, kwds=kwargs
                )

        distro = distro_cls(*args, **kwargs)

        if rvid is None:
            return RandomVariable(distro, size=size)
        else:
            return RandomVariable(distro, size=size, rvid=rvid)

    return _inner


def wrap(distro_cls, *args, **kwargs):
    """Wrap a SciPy Stats distribution with rvmath class"""

    return builder(distro_cls)(*args, **kwargs)


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

    def random_vars(self) -> ty.Generator[ty.Tuple[str, stats.rv_continuous]]:
        """Yields all random variables and their distributions within this expression.

        Yields
        ------
        str, stats.rv_continuous
            variable name, distribution
        """

        for rvid, obj in self.random_objs():
            yield rvid, (obj.distro, obj.size)

    def random_objs(self) -> ty.Generator[ty.Tuple[str, RandomVariable]]:
        """Yield all random rvmath object within this expression.

        Yields
        ------
        str, RandomVariable

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

    def draw(
        self, size=1, random_state=None
    ) -> ty.Dict[str, np.ndarray or numbers.Number]:
        """Draw values for the random variables within this expression."""

        robjs = dict(self.random_objs())

        # We first evaluate the non-dependent distributions.
        realization = {
            rvid: obj.distro.rvs(combine_size(obj.size, size), random_state)
            for rvid, obj in self.random_objs()
            if not isinstance(obj, DependentRandomVariable)
        }

        # Then we build a dependency graph.
        deps = {
            rvid: set(_rvid for _rvid, _ in obj.children_random_objs())
            for rvid, obj in robjs.items()
            if isinstance(obj, DependentRandomVariable)
        }

        for layer in solve_dependencies(deps):
            for rvid in layer:
                cur = robjs[rvid]
                sz = combine_size(cur.size, size)
                if isinstance(cur, DependentRandomVariable):
                    realization[rvid] = cur.freeze(realization).rvs(sz, random_state)
                else:
                    realization[rvid] = cur.distro.rvs(sz, random_state)

        return realization

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

    def random_objs(self):
        yield from super().random_objs()
        yield from self.children_random_objs()

    def children_random_objs(self):
        for arg in self.args:
            if isinstance(arg, RVMixin):
                yield from arg.random_objs()

        for k, v in self.kwds.items():
            if isinstance(v, RVMixin):
                yield from v.random_objs()

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

    def random_objs(self):
        yield self.rvid, self

    def eval(self, realization):
        if self.rvid in realization:
            return realization[self.rvid]
        return self.distro()

    def __str__(self):
        obj = self.distro
        s = tuple((str(a) for a in obj.args)) + tuple(
            (f"{k}= {v}" for k, v in obj.kwds)
        )
        return f"{obj.dist.name}({', '.join(s)})#{self.rvid}"


@dataclass(frozen=True)
class DependentRandomVariable(WithArg, RandomVariable):
    """A random variable that depends on other random variables
    (e.g. it's mean value is drawn from another random variable).
    """

    def eval(self, realization):
        return realization[self.rvid]

    def freeze(self, realization):
        args, kwds = self.get_args_kwds(realization)
        return self.distro(*args, **kwds)

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

    def random_objs(self):
        if isinstance(self.value, RVMixin):
            yield from self.value.random_objs()

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

    def random_objs(self):
        if isinstance(self.value1, RVMixin):
            yield from self.value1.random_objs()

        if isinstance(self.value2, RVMixin):
            yield from self.value2.random_objs()

    def eval(self, realization):
        return self.op(
            eval_value(self.value1, realization),
            eval_value(self.value2, realization),
        )

    def __str__(self):
        return str(self.value1) + " " + _OP_STR[self.op] + " " + str(self.value2)


One = UnaryOp(operator.pos, 1)


def solve_dependencies(dependencies):
    """Solve a dependency graph.

    Parameters
    ----------
    dependencies :
        dependency dictionary. For each key, the value is an iterable indicating its
        dependencies.

    Returns
    -------
    type
        iterator of sets, each containing keys of independents tasks dependent only of
        the previous tasks in the list.

    """
    while dependencies:
        # values not in keys (items without dep)
        t = {i for v in dependencies.values() for i in v} - dependencies.keys()
        # and keys without value (items without dep)
        t.update(k for k, v in dependencies.items() if not v)
        # can be done right away
        if not t:
            raise ValueError(
                "Cyclic dependencies exist among these items: {}".format(
                    ", ".join(repr(x) for x in dependencies.items())
                )
            )
        # and cleaned up
        dependencies = {k: v - t for k, v in dependencies.items() if v}
        yield t
