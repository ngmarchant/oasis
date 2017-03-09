import numpy as np
import warnings
from scipy.special import expit

from .base import (BaseSampler, verify_proba)

class ImportanceSampler(BaseSampler):
    """
    Implements importance sampling to estimate alpha-weighted F-measures.

    For details of the method, see the following paper:

        C. Sawade, N. Landwehr, and T. Scheffer, “Active Estimation of
        F-Measures,” in Advances in Neural Information Processing Systems 23,
        2010, pp. 2083–2091

    Input
    -----
    labels : int numpy array of length n
        array containing binary labels (assumed to be `0` or `1`) for each
        data point.

    probs : float numpy array of length n
        array containing positive class probabilities p(1|x) for each data
        point.

    preds : int numpy array of length n
        array containing predicted binary labels for each data point.

    alpha : float, optional, default 0.5
        weight to use for the F-measure. It should be between 0 and 1, with a
        value of 1 corresponding to "precision", a value of 0 corresponding to
        "recall" and a value of 0.5 corresponding to the balanced F-measure
        (equal weight on precision and recall). Note that this parameterisation
        of the weighted F-measure is different to the usual one (see the paper
        cited above for the definition).

    max_iter : int, optional, default None
        maximum number of iterations permitted (used to pre-allocate arrays for
        storing the sampling history). Defaults to the number of data points.

    epsilon : float, optional, default 1e-3
        epsilon-greedy parameter. Takes a value on the closed unit interval.
        The "optimal" distribution is used with probability `1 - epsilon`, and
        the passive distribution is used with probability `epsilon`. The
        sampling is close to "optimal" for small epsilon.

    debug : bool, optional, default False
        if True, prints debugging information.
    """
    def __init__(self, alpha, oracle, predictions, scores, max_iter,
                 proba=False, epsilon = 1e-3, indices = None, debug=False):
        super(ImportanceSampler, self).__init__(alpha, oracle, predictions,
                                          max_iter, indices, debug)
        self.scores = scores
        self.proba = verify_proba(scores, proba)
        self.epsilon = epsilon

        # Map scores to [0,1] interval (if not already probabilities)
        self._probs = self.scores if proba else expit(self.scores)
        self._F_guess = self._calc_F_guess(self.alpha, self.predictions,
                                           self._probs)

        self.inst_pmf = np.empty(self._num_items, dtype=float)
        self._initialise_pmf()

    def _sample_item(self):
        """
        Samples an item according to the instrumental distribution with
        replacement.
        """
        loc = np.random.choice(self._num_items, p = self.inst_pmf)
        weight = (1/self._num_items)/self.inst_pmf[loc]
        return loc, weight, {}

    def _calc_F_guess(self, alpha, predictions, probabilities):
        """
        Calculates an estimate of the F-measure based on the probabilities
        """
        num = np.sum(self._probs * self.predictions)
        den = np.sum(self._probs * (1 - self.alpha) + \
                        self.alpha * self.predictions)
        F_guess = 0.5 if den==0 else num/den
        return F_guess

    def _initialise_pmf(self):
        """
        Calculates the asymptotically "optimal" instrumental distribution.
        """
        # Easy vars
        epsilon = self.epsilon
        alpha = self.alpha
        predictions = self.predictions
        p1 = self._probs
        p0 = 1 - p1
        num_items = self._num_items
        F = self._F_guess

        # Predict positive pmf
        pp_pmf = np.sqrt(alpha**2 * F**2 * p0 + (1 - F)**2 * p1)

        # Predict negative pmf
        pn_pmf = (1 - alpha) * F * np.sqrt(p1)

        # Calculate "optimal" pmf
        self.inst_pmf = ( (predictions == 1) * pp_pmf
                          + (predictions == 0) * pn_pmf )

        # Normalise
        self.inst_pmf = self.inst_pmf/np.sum(self.inst_pmf)

        # Epsilon-greedy version
        self.inst_pmf = ( epsilon * np.repeat(1/num_items, num_items)
                          + (1 - epsilon) * self.inst_pmf )
