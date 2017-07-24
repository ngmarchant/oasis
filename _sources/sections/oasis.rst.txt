.. _oasis:

=====
OASIS
=====
The OASIS method generates estimates of the weighted F-measure for a binary
classifier with respect to a finite pool (test set). In terms of Type I and II
errors, the weighted F-measure is defined as follows:

.. math::

    F_\alpha = \frac{\mathrm{TP}}{\alpha (\mathrm{TP} + \mathrm{FP}) + (1-\alpha)(\mathrm{TP} + \mathrm{FN})}

where :math:`\alpha \in {[0,1]}` is the weight, TP is the number of true
positives, FP is the number of false positives and FN is the number of false
negatives. The weighted F-measure takes on special values for
:math:`\alpha = 0, 0.5, 1`. :math:`F_0` corresponds to recall, :math:`F_{0.5}`
corresponds to F1-score (balanced F-measure), and :math:`F_1` corresponds to
precision.

.. note::
    This section serves as an introductory guide to the functionality of OASIS.
    Please consult the API reference for the :class:`oasis.OASISSampler` class
    for more detailed information.

Required input
==============
OASIS requires four main inputs:

#. ``alpha``: the F-measure weight as defined above
#. ``predictions``: predicted labels for the items in the pool (according to the
   classifier(s))
#. ``scores``: classifier scores for the items in the pool (e.g. estimated
   positive class probability, distance from decision boundary)
#. ``oracle``: a function which returns ground truth labels for items in the
   pool (i.e. an interface to a labeller)

Multiple classifiers
--------------------
OASIS supports evaluating multiple classifiers on the same test set
*in parallel*. It will optimise the sampling strategy to ensure that the
performance estimates for all of the classifiers have minimal variance.
To use this functionality, simply pass two-dimensional arrays for the
``predictions`` and ``scores`` inputs. The rows of these arrays should
correspond to the items and the columns should correspond to the different
classifiers.

Item identifiers
----------------
OASIS permits the use of custom unique identifiers for the items in the pool.
Simply pass an array to the ``identifiers`` argument. If custom identifiers are
provided, they will be used when making queries to the oracle function.
Otherwise OASIS will refer to the items in the pool according to their index
in the input ``predictions``/``scores`` arrays.

Other parameters
================
There are several optional parameters that may be used to control the
behaviour of OASIS. They are explained in detail in the API reference for the
:class:`oasis.OASISSampler` class. Here, we focus on two important sets of
optional parameters: those that control the stratification of the pool and
those that control the greediness of the sampling.

Stratification parameters
-------------------------
The OASIS method involves partitioning the pool into non-overlapping sets
called strata. The strata should be roughly homogeneous---i.e. they should
contain items with roughly the same labels (or label probabilities if the
labels are uncertain). By default, OASIS will attempt to stratify the pool
based on the scores using the 'cum_sqrt_F' method. It will also attempt to
choose an appropriate number of strata automatically, based on the distribution
of the scores.

On occassion, the strata chosen by the automatic method are inappropriate. For
example, if the score distribution is highly skewed, the automatic method may
yield some strata which are too small (containing only a couple of items). For
this reason, it is always advisable to check that the strata are sensible
before beginning labelling.

The strata can be tweaked as follows:

* by providing a custom :class:`oasis.stratification.Strata` instance for the
  ``strata`` argument
* by providing the ``stratification_method``, ``stratification_n_strata``
  and/or ``stratification_n_bins`` keyword arguments (to override the automatic
  values)

Greediness parameters
---------------------
OASIS samples from the "optimal" instrumental distribution with probability
``1 - epsilon``, and from the passive (uniform) distribution with
probability ``epsilon``. The default setting for the greediness parameter
is ``epsilon = 1e-3``---i.e. it is set to be greedy for more rapid convergence.
To conduct more "explorative" sampling, you should set ``epsilon`` closer to 1.

There are other parameters which indirectly control the greediness of the
sampling. The ``decaying_prior`` parameter is set to ``True`` by default and
iteratively decreases the reliance on the prior in favour of the sampled
labels, which is effectively a greedy approach. A related parameter is
``prior_strength``, which quantifies the initial weight of the prior. If
``prior_strength`` is small (close to zero), then the prior will be weak in
comparison to the sampled labels, which is also a greedy strategy.

Labelling and estimation
========================
Once an instance of :class:`oasis.OASISSampler` is initialised, labelling can
begin. To sample and label a sequence of items, simply call the ``sample``
method. In between calls of the ``sample`` method, you may want to check the
estimate of the F-measure to see whether it is converging. The history of
estimates (for each iteration) is stored in the ``estimate_`` attribute.

.. note::
    There exists an alternative method to ``sample`` called
    ``sample_distinct``, which is useful when sampling with replacement.
    This method continues to sample from the pool until a given number of
    **distinct** items (i.e. previously unlabelled items) have been sampled.
    This differs from ``sample``, which doesn't distinguish between previously
    labelled/unlabelled items.
