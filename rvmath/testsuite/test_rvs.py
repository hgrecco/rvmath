import operator

import numpy as np
import pytest
import scipy.stats as stats
from numpy import testing as npt

import rvmath as rvm
import rvmath.base as pb

binops = (operator.add, operator.sub, operator.mul, operator.truediv, operator.pow)


def test_numbers():
    assert pb.UnaryOp(operator.pos, 3).rvs(1) == 3
    assert pb.UnaryOp(operator.neg, 3).rvs(1) == -3

    assert pb.BinaryOp(operator.add, 3, 1).rvs(1) == 4
    assert pb.BinaryOp(operator.sub, 3, 1).rvs(1) == 2
    assert pb.BinaryOp(operator.mul, 3, 2).rvs(1) == 6
    assert pb.BinaryOp(operator.truediv, 3, 2).rvs(1) == 3 / 2
    assert pb.BinaryOp(operator.pow, 3, 2).rvs(1) == 3 ** 2


@pytest.mark.parametrize("op", binops)
def test_distro(op):
    ref = op(stats.uniform(0, 1).rvs(100, random_state=1234), 4)
    fcalc = op(rvm.uniform(0, 1), 4).rvs(100, random_state=1234)
    npt.assert_equal(ref, fcalc)


def test_np_sum():
    ref_val = np.sum(stats.uniform(0, 1).rvs(100, random_state=1234))
    calc_val = np.sum(rvm.uniform(0, 1)).rvs(100, random_state=1234)
    assert ref_val == calc_val

    ref_val = np.sum(stats.uniform(0, 1).rvs((10, 10), random_state=1234))
    calc_val = np.sum(rvm.uniform(0, 1)).rvs((10, 10), random_state=1234)
    assert ref_val == calc_val

    ref_val = np.sum(stats.uniform(0, 1).rvs((10, 10), random_state=1234), axis=1)
    calc_val = np.sum(rvm.uniform(0, 1), axis=1).rvs((10, 10), random_state=1234)
    npt.assert_equal(ref_val, calc_val)


def test_np_cos():
    ref_val = np.cos(stats.uniform(0, 1).rvs(100, random_state=1234))
    calc_val = np.cos(rvm.uniform(0, 1)).rvs(100, random_state=1234)
    npt.assert_equal(ref_val, calc_val)

    ref_val = np.cos(stats.uniform(0, 1).rvs((10, 10), random_state=1234))
    calc_val = np.cos(rvm.uniform(0, 1)).rvs((10, 10), random_state=1234)
    npt.assert_equal(ref_val, calc_val)
