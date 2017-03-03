import numpy as np
from scipy.special import expit
import copy
import warnings

class Druck:
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
    def __init__(self, labels, strata, alpha=0.5, replace=True,
                 max_iter = None, debug=False):
        self._original_strata = strata
        self.labels = labels
        self.strata = copy.deepcopy(strata)
        self.alpha = alpha
        self.replace = replace
        self._max_iter = max_iter
        self.debug = debug

        self.t = 0

        if (not self.replace) or (self._max_iter is None):
            self._max_iter = self.strata.num_pts

        # Terms used to calculate F-measure (we update them iteratively)
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0

        self.n_sampled = np.zeros(self.strata.num_st, dtype=int)

        # Array to store history of F-measure estimates
        self.F = np.repeat(np.nan, self._max_iter)

        # Array to record whether oracle was queried at each iteration
        self.queried_oracle = np.repeat(False, self._max_iter)

    def _update_F_terms(self, y, yhat):
        """
        Iteratively update the terms that are used to calculate the F-measure
        after a new point is sampled with label `y` and prediction `yhat`.
        """
        if y == 1 and yhat == 1:
            # Point is true positive
            self._TP_term = self._TP_term + 1
            self._PP_term = self._PP_term + 1
            self._P_term = self._P_term + 1
        elif yhat == 1:
            # Point is false positive
            self._PP_term = self._PP_term + 1
        elif y == 1:
            # Point is false negative
            self._P_term = self._P_term + 1

    def _update_F(self):
        """
        Records the latest estimate of the F-measure
        """

        t = self.t

        num = self._TP_term
        den = (self.alpha * self._PP_term + (1 - self.alpha) * self._P_term)

        if den == 0:
            self.F[t] = np.nan
        else:
            self.F[t] = num/den

    def _query_label(self, sample_id):
        """
        Queries the oracle for the label of the datapoint with id `sample_id`.
        Also records that the oracle was queried.

        Returns the ground truth label `0` or `1`.
        """
        t = self.t

        # Get label
        y = self.labels[sample_id]

        # Record that label was queried in this iteration
        self.queried_oracle[t] = True

        return y

    def reset(self):
        self.t = 0

        self.strata = copy.deepcopy(self._original_strata)

        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0

        self.n_sampled = np.zeros(self.strata.num_st, dtype=int)

        self.F = np.repeat(np.nan, self._max_iter)

        self.queried_oracle = np.repeat(False, self._max_iter)

    def sample(self, n_iter):
        """
        Samples `n_iter` points
        fixed_stratum :

        """
        t_i = self.t
        t_f = n_iter + self.t

        assert t_f <= self.F.shape[0]

        for t in range(t_i, t_f):
            # Check if there are any more points to sample
            if (not self.replace and
                    np.sum(self.strata.populations - self.n_sampled) == 0):
                print("All points have been sampled")
                return

            # Sample label and record weight
            stratum_idx, sample_loc = self.strata.sample(prob_dist = None, replace = self.replace)

            # Check if label has already been queried and stored
            y = self.strata.labels[stratum_idx][sample_loc]
            if np.isnan(y):
                # Need to query oracle
                sample_idx = self.strata.allocations[stratum_idx][sample_loc]
                y = self._query_label(sample_idx)
                # Store label
                self.strata.update_label(stratum_idx, sample_loc, y)

            # Get prediction
            pred = self.strata.preds[stratum_idx][sample_loc]

            if self.debug == True:
                print("TP_term: {}, PP_term: {}, P_term: {}".format(self._TP_term, self._PP_term, self._P_term))
                print("Sampled label {} for point {} in stratum {}.".format(y,self.strata.allocations[stratum_idx][sample_loc], stratum_idx))

            self._update_F_terms(y, pred)
            self._update_F()
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
