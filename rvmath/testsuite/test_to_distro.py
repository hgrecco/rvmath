import pytest
import scipy.stats as stats
from numpy import testing as npt

import rvmath as rvm


def check_distribution(rvs, cdf):
    alpha = 0.01
    st, pval = stats.kstest(rvs, cdf, N=1_000)
    if pval < alpha:
        D, pval = stats.kstest(rvs, cdf, N=1_000)
        npt.assert_(
            pval > alpha,
            "D = " + str(D) + "; pval = " + str(pval) + "; alpha = " + str(alpha),
        )


def check_discrete_distribution(rvs1, rvs2):
    alpha = 0.01
    N = 1_000
    st, _, pval = stats.anderson_ksamp([rvs1(N), rvs2(N)])
    if pval < alpha:
        D, _, pval = stats.anderson_ksamp([rvs1(N), rvs2(N)])
        npt.assert_(
            pval > alpha,
            "D = " + str(D) + "; pval = " + str(pval) + "; alpha = " + str(alpha),
        )


@pytest.mark.parametrize(
    "expr,ref",
    [
        (rvm.uniform(0, 1), stats.uniform(0, 1)),
        (rvm.uniform(0, 1) + 1, stats.uniform(1, 1)),
        (rvm.uniform(0, 1) - 1, stats.uniform(-1, 1)),
        (rvm.uniform(0, 1) * 2, stats.uniform(0, 2)),
        (rvm.uniform(0, 1) / 2, stats.uniform(0, 0.5)),
        (+rvm.uniform(0, 1), stats.uniform(0, 1)),
        (-rvm.uniform(0, 1), stats.uniform(-1, 1)),
    ],
)
def test_to_distro(expr, ref):
    distro = expr.to_distro("dummy")

    assert hasattr(distro, "pdf")

    check_distribution(ref.rvs, ref.cdf)
    check_distribution(expr.rvs, distro.cdf)
    check_distribution(distro.rvs, distro.cdf)
    check_distribution(distro.rvs, ref.cdf)
    check_distribution(expr.rvs, distro.cdf)


@pytest.mark.parametrize(
    "expr,ref",
    [
        (rvm.poisson(10), stats.poisson(10)),
    ],
)
def test_to_distro_discrete(expr, ref):
    distro = expr.to_distro("dummy", discrete=True)

    assert hasattr(distro, "pmf")

    check_discrete_distribution(ref.rvs, ref.rvs)
    check_discrete_distribution(expr.rvs, distro.rvs)
    check_discrete_distribution(distro.rvs, distro.rvs)
    check_discrete_distribution(distro.rvs, ref.rvs)
    check_discrete_distribution(expr.rvs, distro.rvs)
