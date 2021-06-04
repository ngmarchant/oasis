=====================================================
OASIS: a tool for efficient evaluation of classifiers
=====================================================

.. image:: https://travis-ci.org/ngmarchant/oasis.svg?branch=master
    :target: https://travis-ci.org/ngmarchant/oasis
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
.. image:: https://badge.fury.io/py/oasis.svg
    :target: https://pypi.python.org/pypi/oasis

Overview
========
OASIS is a tool for evaluating binary classifiers when ground truth class
labels are not immediately available, but can be obtained at some cost (e.g.
by asking human annotators). The tool takes an unlabelled test set as input and
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
  (examples include entity resolution, information retrieval, text
  classification and many problems in the medical domain)

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

    N. G. Marchant and B. I. P. Rubinstein, “In Search of an Entity Resolution
    OASIS: Optimal Asymptotic Sequential Importance Sampling,” in 
    *Proceedings of the VLDB Endowment*, vol. 10, no. 11, pp. 1322-1333, 2017.
    `link <http://www.vldb.org/pvldb/vol10/p1322-rubinstein.pdf>`_

License and disclaimer
======================
OASIS is released under the MIT license. Please see the LICENSE file included
with the source.

The software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.

.. toctree::
    :maxdepth: 2
    :hidden:

    sections/installation.rst
    sections/oasis.rst
    sections/other-samplers.rst
    sections/api-ref.rst
    tutorial/tutorial.ipynb
