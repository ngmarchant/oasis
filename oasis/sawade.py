import numpy as np
import warnings
from scipy.special import expit

from .base import (BaseSampler, verify_proba)

class ImportanceSampler(BaseSampler):
    """Importance sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an instrumental
    distribution that minimises asymptotic variance. The instrumental
    distribution is estimated based on classifier confidence scores. True
    labels are queried from an oracle. See reference [Sawade2010]_ for details.

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

    proba : bool, optional, default False
        indicates whether the scores are probabilistic, i.e. on the interval
        [0, 1].

    epsilon : float, optional, default 1e-3
        epsilon-greedy parameter. Valid values are on the interval [0, 1]. The
        "asymptotically optimal" distribution is sampled from with probability
        `1 - epsilon` and the passive distribution is sampled from with
        probability `epsilon`. The sampling is close to "optimal" for small
        epsilon.

    max_iter : int, optional, default None
        space for storing the sampling history is limited to a maximum
        number of iterations. Once this limit is reached, sampling can no
        longer continue. If no value is given, defaults to the size of
        the pool.

    replace : bool, optional, default True
        whether to sample with or without replacement.

    Other Parameters
    ----------------
    indices : array-like, optional, default None
        ordered array of unique identifiers for the items in the pool.
        Should match the order of the "predictions" parameter. If no value is
        given, defaults to [0, 1, ..., pool_size].

    debug : bool, optional, default False
        whether to print out verbose debugging information.

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

    inst_pmf_ : numpy.ndarray, shape=(pool_size,)
        epsilon-greedy instrumental pmf used for sampling.

    [Sawade2010] C. Sawade, N. Landwehr, and T. Scheffer, “Active Estimation of
    F-Measures,” in Advances in Neural Information Processing Systems 23, 2010,
    pp. 2083–2091
    """
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 epsilon=1e-3, max_iter=None, indices = None, debug=False):
        super(ImportanceSampler, self).__init__(alpha, predictions, oracle,
                                          max_iter, indices, debug)
        self.scores = scores
        self.proba = verify_proba(scores, proba)
        self.epsilon = epsilon

        # Map scores to [0,1] interval (if not already probabilities)
        self._probs = self.scores if proba else expit(self.scores)
        self._F_guess = self._calc_F_guess(self.alpha, self.predictions,
                                           self._probs)

        self.inst_pmf_ = np.empty(self._pool_size, dtype=float)
        self._initialise_pmf()

    def _sample_item(self):
        """Sample an item from the pool according to the instrumental
        distribution
        """
        loc = np.random.choice(self._pool_size, p = self.inst_pmf_)
        weight = (1/self._pool_size)/self.inst_pmf_[loc]
        return loc, weight, {}

    def _calc_F_guess(self, alpha, predictions, probabilities):
        """Calculate an estimate of the F-measure based on the scores"""
        num = np.sum(self._probs * self.predictions)
        den = np.sum(self._probs * (1 - self.alpha) + \
                        self.alpha * self.predictions)
        F_guess = 0.5 if den==0 else num/den
        return F_guess

    def _initialise_pmf(self):
        """Calculate the epsilon-greedy instrumental distribution"""
        # Easy vars
        epsilon = self.epsilon
        alpha = self.alpha
        predictions = self.predictions
        p1 = self._probs
        p0 = 1 - p1
        pool_size = self._pool_size
        F = self._F_guess

        # Predict positive pmf
        pp_pmf = np.sqrt(alpha**2 * F**2 * p0 + (1 - F)**2 * p1)

        # Predict negative pmf
        pn_pmf = (1 - alpha) * F * np.sqrt(p1)

        # Calculate "optimal" pmf
        self.inst_pmf_ = ( (predictions == 1) * pp_pmf
                          + (predictions == 0) * pn_pmf )

        # Normalise
        self.inst_pmf_ = self.inst_pmf_/np.sum(self.inst_pmf_)

        # Epsilon-greedy version
        self.inst_pmf_ = ( epsilon * np.repeat(1/pool_size, pool_size)
                          + (1 - epsilon) * self.inst_pmf_ )
