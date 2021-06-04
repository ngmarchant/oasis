.. _other-samplers:

==============
Other samplers
==============
The OASIS package also implements three alternative sampling-based estimation
methods. These alternative methods are all non-adaptive---i.e. they don't learn
from the labels as they are received from the oracle. Experiments comparing
these alternative methods with OASIS in the entity resolution domain are
presented in [Marchant17]_.

.. _passive-sampler:

Passive sampler
===============
This method is the simplest approach and involves choosing items to label by
sampling uniformly from the pool. It supports sampling with or without
replacement through the ``replace`` parameter.

For further information, see the :class:`oasis.PassiveSampler` class.

.. _importance-sampler:

Non-adaptive importance sampler
===============================
This method is based on importance sampling using a similar "optimal"
instrumental distribution to the one used in OASIS (see [Sawade09]_). The
key difference is that this method approximates the "optimal" instrumental
distribution based solely on the classifier scores, which may be inaccurate.
It cannot adapt based on the incoming ground truth labels.

For further information, see the :class:`oasis.ImportanceSampler` class.

.. _stratified-sampler:

Stratified sampler
==================
This method is based on unbiased stratified sampling (see [Druck11]_). Like
OASIS, it depends on a partitioning of the pool into a collection of strata. In
practice, it tends to perform similarly to passive sampling.

For further information, see the :class:`oasis.DruckSampler` class.

References
----------
.. [Druck11]  G. Druck and A. McCallum, “Toward Interactive Training and
   Evaluation,” in *Proceedings of the 20th ACM International Conference on
   Information and Knowledge Management*, pp. 947–956, 2011.
.. [Sawade09] C. Sawade, N. Landwehr, and T. Scheffer, “Active Estimation of
   F-Measures,” in *Advances in Neural Information Processing Systems*, pp.
   2083–2091, 2010.
.. [Marchant17] N. G. Marchant and B. I. P. Rubinstein, “In Search of an Entity
    Resolution OASIS: Optimal Asymptotic Sequential Importance Sampling,” in
    *Proceedings of the VLDB Endowment*, vol. 10, no. 11, pp. 1322-1333, 2017.
