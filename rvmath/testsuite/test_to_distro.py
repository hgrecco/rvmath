import pytest
import scipy.stats as stats

import rvmath as rvm


def check_distribution(rvs, cdf):
    st, pval = stats.kstest(rvs, cdf, N=1_000)
    assert pval > 0.01


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

    check_distribution(expr.rvs, distro.cdf)
    check_distribution(distro.rvs, distro.cdf)
    check_distribution(distro.rvs, ref.cdf)
    check_distribution(expr.rvs, distro.cdf)
