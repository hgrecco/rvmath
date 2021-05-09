import operator

import rvmath.base as pb


def test_str_binop():
    assert str(pb.BinaryOp(operator.add, 1, 4)) == "1 + 4"
    assert str(pb.BinaryOp(operator.sub, 1, 4)) == "1 - 4"
    assert str(pb.BinaryOp(operator.mul, 1, 4)) == "1 * 4"
    assert str(pb.BinaryOp(operator.truediv, 1, 4)) == "1 / 4"
    assert str(pb.BinaryOp(operator.pow, 1, 4)) == "1 ** 4"


def test_str_unop():
    assert str(pb.UnaryOp(operator.pos, 1)) == "+1"
    assert str(pb.UnaryOp(operator.neg, 1)) == "-1"
