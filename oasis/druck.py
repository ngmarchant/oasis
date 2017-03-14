import numpy as np
import warnings

from .base import BaseSampler
from .passive import PassiveSampler
from .stratification import auto_stratify

class DruckSampler(PassiveSampler):
    """Stratified sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an adaptive instrumental
    distribution that minimises asymptotic variance. See reference
    [Druck2011]_ for details.

    Parameters
    ----------
    alpha : float
        weight for the F-measure. Valid weights are on the interval [0, 1].
        ``alpha == 1`` corresponds to precision, ``alpha == 0`` corresponds to
        recall, and ``alpha == 0.5`` corresponds to the balanced F-measure.

    predictions : array-like, shape=(pool_size,)
        ordered array of predicted labels for each item in the pool. Valid
        labels are "0" or "1".

    scores : array-like, shape=(pool_size,)
        ordered array of scores which quantify the classifier confidence for
        the items in the pool. High scores indicate a high confidence that
        the true label is a "1" (and vice versa for label "0").

    oracle : function
        a function which takes an item id as input and returns the item's true
        label. Valid labels are "0" or "1".

    strata : Strata instance, optional, default None
        describes how to stratify the pool. If not given, the stratification
        will be done automatically based on the scores given. Additional
        keyword arguments may be passed to control this automatic
        stratification (see below).

    max_iter : int, optional, default None
        space for storing the sampling history is limited to a maximum
        number of iterations. Once this limit is reached, sampling can no
        longer continue. If no value is given, defaults to the size of
        the pool.

    replace : boolean
        If True, sample with replacement, otherwise, sample without
        replacement.

    Other Parameters
    ----------------
    indices : array-like, optional, default None
        ordered array of unique identifiers for the items in the pool.
        Should match the order of the "predictions" parameter. If no value is
        given, defaults to [0, 1, ..., pool_size].

    debug : bool, optional, default False
        if True, prints debugging information.

    Attributes
    ----------
    estimate_ : numpy.ndarray
        array of F-measure estimates at each iteration. Iterations that yield
        an undefined estimate (e.g. 0/0) are recorded as NaN values.

    queried_oracle_ : numpy.ndarray
        array of bools which records whether the oracle was queried at each
        iteration (True) or whether a cached label was used (False).

    cached_labels_ : numpy.ndarray, shape=(pool_size,)
        ordered array of true labels for the items in the pool. The order
        matches that used for the "predictions" parameter. Items which have not
        had their labels queried are recorded as NaNs.

    t_ : int
        iteration index.

    References
    ----------
    .. [Druck2011] G. Druck and A. McCallum, "Toward Interactive Training and
       Evaluation," in Proceedings of the 20th ACM International Conference on
       Information and Knowledge Management, 2011, pp. 947–956.

    """
    def __init__(self, alpha, predictions, scores, oracle, strata=None,
                 max_iter=None, indices=None, replace=True, debug=False,
                 **kwargs):
        super(DruckSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, indices, replace, debug)
        self.scores = scores
        self.strata = strata

        # Generate strata if not given
        if self.strata is None:
            self.strata = auto_stratify(self.scores, **kwargs)

    def _sample_item(self):
        """Sample an item from the strata
        """
        # Sample label and record weight
        loc, stratum_idx = self.strata.sample(pmf = self.strata.weights_)

        return loc, 1, {'stratum': stratum_idx}

    def reset(self):
        """Resets the sampler to its initial state

        Note
        ----
        This will destroy the label cache and history of estimates
        """
        super(DruckSampler, self).reset()
        self.strata.reset()
