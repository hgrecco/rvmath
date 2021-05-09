.. image:: https://img.shields.io/pypi/v/rvmath.svg
    :target: https://pypi.python.org/pypi/rvmath
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/l/rvmath.svg
    :target: https://pypi.python.org/pypi/rvmath
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/rvmath.svg
    :target: https://pypi.python.org/pypi/rvmath
    :alt: Python Versions

.. image:: https://github.com/hgrecco/rvmath/workflows/CI/badge.svg?branch=main
    :target: https://github.com/hgrecco/rvmath/actions?query=workflow%3ACI

.. image:: https://github.com/hgrecco/rvmath/workflows/Lint/badge.svg?branch=main
    :target: https://github.com/hgrecco/rvmath/actions?query=workflow%3ALint

.. image:: https://coveralls.io/repos/github/hgrecco/rvmath/badge.svg?branch=main
    :target: https://coveralls.io/github/hgrecco/rvmath?branch=main


rvmath: math with random variables, the easy way
================================================

`rvmath` is a Python package to build and evaluate
mathematical expressions involving random variables.

Do you want to draw 10 values from a distribution resulting
from `a * cos(b + c)` where `a ~ Poisson`, `b ~ Uniform`,
and `c ~ Normal`? No problem:

    >>> import rvmath as rvm
    >>> z = rvm.poisson(mu=5) * np.cos(rvm.uniform() + rvm.norm())
    >>> z.rvs(10)

It runs in Python 3.8+ depending on NumPy_ and SciPy_.
It is licensed under BSD.

It is extremely easy and natural to use:

.. code-block:: python

    >>> import rvmath as rvm
    >>> x = rvm.uniform()
    >>> y = rvm.uniform()
    >>> z = x - y
    >>> z.rvs(3)  #doctest: +SKIP
    [ 0.56791289 -0.1547692  -0.73984907]
    >>> z.rvs(3)  #doctest: +SKIP
    [-0.33095289 -0.08664128  0.09938225]

Briefly, `x` and `y` are random variables drawn from a uniform distribution.
`z` is a random variable drawn from a distribution obtained by subtracting
two uniform distributions. `z.rvs(3)` draw 3 values from such distribution.

Behind the scenes, `probalc` calls `SciPy Stats`_ to generate random variates
of all random variables and perform all necessary calculations.

`rvmath` builds upon `Scipy Stats`_ and therefore all continuous distributions
available there are also here, with the same name and arguments. `rvs` also follows
the same API, namely:

    - **size**: int or tuple of ints, optional
      Defining number of random variates (default is 1).
    - **random_state**: None, int, RandomState, Generator, optional
      If seed is None the RandomState singleton is used. If seed is an int,
      a new RandomState instance is used, seeded with seed. If seed is already
      a RandomState or Generator instance, then that object is used. Default is None.

An important feature is that random variables have an identity and therefore
the following code gives the expected result.

.. code-block:: python

    >>> w = x - x
    >>> w.rvs(3)
    [0., 0., 0.]

You can also use NumPy functions.

.. code-block:: python

    >>> c = np.cos(x)
    >>> c.rvs(3)


Finally, you can convert the expression into a SciPy distribution:

.. code-block:: python

    >>> distro = c.to_distro(name="my_distro")

with useful methods such as `rvs`, `pdf`, `cdf`, available.


Quick Installation
------------------

To install rvmath, simply (*soon*):

.. code-block:: bash

    $ pip install rvmath

and then simply enjoy it!


----

rvmath is maintained by a community. See AUTHORS_ for a complete list.

To review an ordered list of notable changes for each version of a project,
see CHANGES_


.. _`NumPy`: http://www.numpy.org/
.. _`SciPy`: http://www.scipy.org/
.. _`SciPy Stats`: https://docs.scipy.org/doc/scipy/reference/stats.html
.. _`pytest`: https://docs.pytest.org/
.. _`AUTHORS`: https://github.com/hgrecco/rvmath/blob/master/AUTHORS
.. _`CHANGES`: https://github.com/hgrecco/rvmath/blob/master/CHANGES
