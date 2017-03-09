import numpy as np
from scipy.special import expit
import copy
import warnings

from .base import BaseSampler

class Druck(BaseSampler):
    """
    Input
    -----
    labels : int numpy array of length n
        array containing binary labels (assumed to be `0` or `1`) for each
        data point.

    strata: an instance of the `Strata` class
        contains information about the data points (allocations, scores)
        and facilitates sampling from the strata.

    alpha : float, optional, default 0.5
        weight to use for the F-measure. It should be between 0 and 1, with a
        value of 1 corresponding to "precision", a value of 0 corresponding to
        "recall" and a value of 0.5 corresponding to the balanced F-measure
        (equal weight on precision and recall). Note that this parameterisation
        of the weighted F-measure is different to the usual one (see the paper
        cited above for the definition).

    replace : boolean
        If True, sample with replacement, otherwise, sample without
        replacement.

    debug : bool, optional, default False
        if True, prints debugging information.
    """
    def __init__(self, alpha, oracle, predictions, scores, strata=None,
                 max_iter=None, indices=None, replace=True, debug=False):
        self.scores = scores
        self._original_strata = strata
        self.strata = copy.deepcopy(strata)
        self.replace = replace
        self.debug = debug

        #TODO construct strata if not given
        #TODO don't require scores if strata is given?

        self.n_sampled = np.zeros(self.strata.num_st, dtype=int)

    def _sample_item(self):
        stratum_idx, sample_loc = self.strata.sample(prob_dist = None, replace = self.replace)

        #TODO convert to appropriate form

        return loc, 1, {'stratum_idx':stratum_idx}

    def reset(self):
        super(PassiveSampler, self).reset()

        self.strata = copy.deepcopy(self._original_strata)
        self.n_sampled = np.zeros(self.strata.num_st, dtype=int)

    def sample(self, n_iter):
        """
        Samples `n_iter` points
        fixed_stratum :

        """

        for t in range(t_i, t_f):
            # Check if there are any more points to sample

            # TODO ensure that these extra steps are taken care of in the BaseSampler
            # Or rewrite to ensure that they can be carried out here.
            if (not self.replace and
                    np.sum(self.strata.populations - self.n_sampled) == 0):
                print("All points have been sampled")
                return

            self.n_sampled[stratum_idx] += 1

            self.t = self.t + 1

    def sample_until(self, n_goal):
        """
        Sample until `n_goal` labels are queried from the oracle
        """

        n_seen = np.sum(self.queried_oracle)

        if n_seen >= n_goal:
            print("Have already queried {} labels from the oracle".format(n_seen))
            return

        if n_goal > self.strata.num_pts + 1:
            print("{} is greater than the number of points in the dataset".format(n_goal))
            return

        while n_seen < n_goal:
            self.sample(1)
            n_seen = n_seen + self.queried_oracle[self.t - 1]*1
