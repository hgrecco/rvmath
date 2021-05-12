import operator

import numpy as np
import pytest
import scipy.stats as stats
from numpy import testing as npt

import rvmath as rvm
from rvmath import base as pb


def lengen(gen):
    return sum(1 for _ in gen)


def test_distro():
    ref = stats.uniform(0, 1)
    calc = rvm.uniform(0, 1)

    assert calc.rvid is not None
    assert calc.size is None
    assert lengen(calc.random_vars()) == 1
    npt.assert_equal(ref.rvs(100, random_state=1234), calc.rvs(100, random_state=1234))


def test_size():
    ref = stats.uniform(0, 1)
    calc = rvm.uniform(0, 1, size=2)
    assert calc.rvid is not None
    assert calc.size == 2
    assert lengen(calc.random_vars()) == 1
    npt.assert_equal(ref.rvs(2, random_state=1234), calc.rvs(random_state=1234))


def test_distro_id():
    ref = stats.uniform(0, 1)
    calc = rvm.uniform(0, 1, rvid="calc")

    assert calc.rvid == "calc"
    assert lengen(calc.random_vars()) == 1
    npt.assert_equal(ref.rvs(100, random_state=1234), calc.rvs(100, random_state=1234))


def test_expression():
    calc = rvm.uniform(0, 1, rvid="calc")
    assert lengen(calc.random_vars()) == 1
    assert lengen(pb.BinaryOp(operator.add, calc, 4).random_vars()) == 1

    assert pb.BinaryOp(operator.add, calc, 4) == calc + 4
    assert pb.BinaryOp(operator.sub, calc, 4) == calc - 4
    assert pb.BinaryOp(operator.mul, calc, 4) == calc * 4
    assert pb.BinaryOp(operator.truediv, calc, 4) == calc / 4
    assert pb.BinaryOp(operator.pow, calc, 4) == calc ** 4

    assert pb.BinaryOp(operator.add, 4, calc) == 4 + calc
    assert pb.BinaryOp(operator.sub, 4, calc) == 4 - calc
    assert pb.BinaryOp(operator.mul, 4, calc) == 4 * calc
    assert pb.BinaryOp(operator.truediv, 4, calc) == 4 / calc
    assert pb.BinaryOp(operator.pow, 4, calc) == 4 ** calc


def test_numpy_function():
    calc = rvm.uniform(0, 1, rvid="calc")
    assert lengen(calc.random_vars()) == 1
    assert lengen(pb.Function(np.sum, (calc,), {}).random_vars()) == 1

    assert pb.Function(np.sum, (calc,), {}) == np.sum(calc)
    assert pb.Function(np.sum, (calc, 1), {}) == np.sum(calc, 1)
    assert pb.Function(np.sum, (calc,), dict(axis=1)) == np.sum(calc, axis=1)


def test_numpy_ufunc():
    calc = rvm.uniform(0, 1, rvid="calc")
    assert lengen(calc.random_vars()) == 1
    assert lengen(pb.Function(np.cos, (calc,), {}).random_vars()) == 1

    assert pb.Function(np.cos, (calc,), {}) == np.cos(calc)
    assert pb.Function(np.cos, (calc,), dict(casting="same")) == np.cos(
        calc, casting="same"
    )


def test_combined_size_and_unsized():
    calc = np.sum(rvm.uniform(size=(3, None), rvid="x33")) * rvm.norm(
        rvid="x"
    ) + rvm.uniform(rvid="y")
    assert lengen(calc.random_vars()) == 3
    d = calc.draw(2)
    assert d["x33"].shape == (3, 2)
    assert d["x"].shape == (2,)
    assert d["y"].shape == (2,)
    assert len(calc.rvs(2)) == 2


@pytest.mark.parametrize(
    "args",
    [
        (None, None),
        (3, None),
        ((3,), None),
        (None, (None,)),
        (3, (None,)),
        ((3,), (None,)),
    ],
)
def test_combine_size_wrong_None(args):
    with pytest.raises(ValueError):
        pb.combine_size(None, None)


def test_combine_size():
    assert pb.combine_size(None, 4) == 4
    assert pb.combine_size(3, 4) == 3
    assert pb.combine_size(3, (4,)) == 3

    assert pb.combine_size((1,), 5) == (1,)
    assert pb.combine_size((None,), 5) == (5,)
    assert pb.combine_size((None, 1), 6) == (6, 1)
    assert pb.combine_size((1, None), 6) == (1, 6)
    assert pb.combine_size((None, None), 6) == (6, 6)
    assert pb.combine_size((1,), (6,)) == (1,)

    with pytest.raises(ValueError):
        assert pb.combine_size((1, None), (6,)) == (1, 6)


def test_combined_distribution():
    calc = np.sum(rvm.uniform(size=(3, 3), rvid="x33")) * rvm.norm(
        rvid="x"
    ) + rvm.uniform(rvid="y")
    assert lengen(calc.random_vars()) == 3
    d = calc.draw(2)
    assert d["x33"].shape == (3, 3)
    assert d["x"].shape == (2,)
    assert d["y"].shape == (2,)
    assert len(calc.rvs(2)) == 2


def test_dependent_rv():
    x = rvm.uniform(rvid="x", size=1)
    y = rvm.norm(loc=x, rvid="y")
    assert isinstance(y, pb.DependentRandomVariable)
    assert set(dict(x.random_vars()).keys()) == {"x"}
    assert set(dict(y.random_vars()).keys()) == {"x", "y"}
    assert len(y.rvs(3)) == 3

    y = rvm.uniform(loc=rvm.uniform(rvid="x"), rvid="y")
    assert set(dict(y.random_vars()).keys()) == {"x", "y"}
    assert len(y.rvs(3)) == 3


def test_dependent_rv_size():
    x = rvm.uniform(rvid="x", size=(2, None))
    y = rvm.norm(loc=x, rvid="y")

    assert isinstance(y, pb.DependentRandomVariable)
    assert set(dict(x.random_vars()).keys()) == {"x"}
    assert set(dict(y.random_vars()).keys()) == {"x", "y"}
    assert y.rvs(3) == (2, 3)

    y = rvm.uniform(loc=rvm.uniform(rvid="x", size=(2, None)), rvid="y")
    assert set(dict(y.random_vars()).keys()) == {"x", "y"}
    assert y.rvs(3) == (2, 3)
