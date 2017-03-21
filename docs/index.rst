=====================================================
OASIS: a tool for efficient evaluation of classifiers
=====================================================

Overview
========
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

When should I use OASIS?
========================
OASIS is particularly useful when:

* you have a test set, but you don't yet have ground truth labels
* ground truth labels can be obtained sequentially (this constraint may be
  relaxed in a future update)
* F1-score, precision, or recall is a sufficient measure for the classifier's
  performance
* the classification problem demonstrates a high degree of class imbalance
  (examples include entity resolution, text classification and many problems in
  the medical domain)

Note that the final point to do with class imbalance does not need to be
satisifed in order for OASIS to provide accurate estimates of classifier
performance. It merely describes when OASIS is expected to excel over simpler
methods, such as uniform sampling (see :ref:`passive-sampler`).

Where can I find out more?
==========================
Details about the interface for OASIS, including the required inputs are given
in the :ref:`oasis` section of this documentation. For more information about
the algorithm itself and a proof of its consistency, please refer to the
following paper:

.. _marchant-17:

    N. G. Marchant and B. I. P. Rubinstein, In Search of an Entity Resolution
    OASIS: Optimal Asymptotic Sequential Importance Sampling,
    arXiv:1703.00617 [cs.LG], Mar 2017.
    `link <https://arxiv.org/pdf/1703.00617.pdf>`_

.. toctree::
    :maxdepth: 2
    :hidden:

    installation.rst
    sections/oasis.rst
    sections/other-samplers.rst
    sections/api-ref.rst
    sections/tutorial.ipynb
