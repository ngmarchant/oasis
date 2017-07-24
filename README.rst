=====
OASIS
=====

.. image:: https://travis-ci.org/ngmarchant/oasis.svg?branch=master
    :target: https://travis-ci.org/ngmarchant/oasis
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
.. image:: https://badge.fury.io/py/oasis.svg
    :target: https://pypi.python.org/pypi/oasis

OASIS is a tool for evaluating binary classifiers when ground truth class
labels are not immediately available, but can be obtained at some cost (e.g.
by asking humans). The tool takes an unlabelled test set as input and
intelligently selects items to label so as to provide a *precise* estimate of
the classifier's performance, whilst *minimising* the amount of labelling
required. The underlying strategy for selecting the items to label is based on
a technique called *adaptive importance sampling*, which is optimised for the
classifier performance measure of interest. Currently, OASIS supports
estimation of the weighted F-measure, which includes the F1-score, precision
and recall.

Important links
===============
Documentation: https://ngmarchant.github.io/oasis

Source: https://www.github.com/ngmarchant/oasis

Technical paper: https://arxiv.org/pdf/1703.00617.pdf

Example
=======
See the Jupyter notebook under ``docs/tutorial/tutorial.ipynb``::

    >>> import oasis
    >>> data = oasis.Data()
    >>> data.read_h5('Amazon-GoogleProducts-test.h5')
    >>> def oracle(idx):
    >>>     return data.labels[idx]
    >>> smplr = oasis.OASISSampler(alpha, data.preds, data.scores, oracle)
    >>> smplr.sample_distinct(5000) #: query labels for 5000 distinct items
    >>> print("Current estimate is {}.".format(smplr.estimate_[smplr.t_ - 1]))


License and disclaimer
======================
The code is released under the MIT license. Please see the LICENSE file for
details.
