import operator

import numpy as np
import scipy.stats as stats
from numpy import testing as nt

import rvmath as rvm
from rvmath import base as pb


def test_distro():
    ref = stats.uniform(0, 1)
    calc = rvm.uniform(0, 1)

    assert calc.rvid is not None
    nt.assert_equal(ref.rvs(100, random_state=1234), calc.rvs(100, random_state=1234))


def test_distro_id():
    ref = stats.uniform(0, 1)
    calc = rvm.uniform(0, 1, rvid="calc")

    assert calc.rvid == "calc"
    nt.assert_equal(ref.rvs(100, random_state=1234), calc.rvs(100, random_state=1234))


def test_expression():
    calc = rvm.uniform(0, 1, rvid="calc")
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
    assert pb.Function(np.sum, (calc,), {}) == np.sum(calc)
    assert pb.Function(np.sum, (calc, 1), {}) == np.sum(calc, 1)
    assert pb.Function(np.sum, (calc,), dict(axis=1)) == np.sum(calc, axis=1)
