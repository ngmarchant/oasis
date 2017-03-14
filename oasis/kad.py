import numpy as np
from scipy.special import expit
import copy
import warnings

from .aoais import BetaBernoulliModel

class Kadane:
    """
    Input
    -----
    labels : int numpy array of length n
        array containing binary labels (assumed to be "0" or "1") for each
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

    n_initial : int, optional, default 0
        sample `n_initial` labels uniformly from each stratum before starting
        the epsilon-greedy strategy.

    frac : double, optional, default 0.01
        make the prior decay to 1/e in strength after a fraction `frac` of the
        total labels are sampled.

    epsilon : float, optional, default 1e-3
        epsilon-greedy parameter. Takes a value on the closed unit interval.
        The "optimal" distribution is used with probability `1 - epsilon`, and
        the passive distribution is used with probability `epsilon`. The
        sampling is close to "optimal" for small epsilon.

    debug : bool, optional, default False
        if True, prints debugging information.
    """
    def __init__(self, labels, strata, alpha=0.5, epsilon=0.001, strength=None,
                 n_init=0, method="full", record_cov_wp = False, debug = False):
        self._original_strata = strata
        self.labels = labels
        self.strata = copy.deepcopy(strata)
        self.alpha = alpha
        self.epsilon = epsilon
        if strength is None:
            self.strength = self.num_st
        else:
            self.strength = strength
        self.n_init = n_init
        self.method = method
        self.record_cov_wp = record_cov_wp
        self.debug = debug

        self.t = 0
        self.num_pts = len(labels)
        self.num_st = self.strata.num_st
        self.n_remaining = copy.deepcopy(self.strata.sizes)
        self.n_sampled = np.zeros(self.num_st, dtype=int)

        # TODO This code calculates the prior -- move to Strata class instead
        if self.strata.calibrated_score:
            self._Y_bar_est = np.array([np.mean(s) for s in self.strata.scores])
            self._T_bar_est = np.array([np.mean(p * self.strata.scores[ix]) for (ix,p) in enumerate(self.strata.preds)])
        else:
            self._Y_bar_est = np.array([np.mean(expit(s - self.strata.score_threshold)) for s in self.strata.scores])
            self._T_bar_est = np.array([np.mean(p * expit(self.strata.scores[ix] - self.strata.score_threshold)) for (ix,p) in enumerate(self.strata.preds)])
        self._P_bar_est = np.array([np.mean(p) for p in self.strata.preds])

        # Instantiate Beta-Bernoulli models for \bar{T}, \bar{P}, \bar{Y}
        #self._weighted_strength = self.strength * self.strata.weights
        self._weighted_strength = self.strength / self.num_st
        self.BB_T = BetaBernoulliModel(
                            alpha_0=self._T_bar_est*self._weighted_strength,
                            beta_0=(1-self._T_bar_est)*self._weighted_strength,
                            size=None, populations=self.strata.sizes,
                            store_var=False, store_wp=True
                            )
        self.BB_P = BetaBernoulliModel(
                            alpha_0=self._P_bar_est*self._weighted_strength,
                            beta_0=(1-self._P_bar_est)*self._weighted_strength,
                            size=None, populations=self.strata.sizes,
                            store_var=False, store_wp=True
                            )
        self.BB_Y = BetaBernoulliModel(
                            alpha_0=self._Y_bar_est*self._weighted_strength,
                            beta_0=(1-self._Y_bar_est)*self._weighted_strength,
                            size=None, populations=self.strata.sizes,
                            store_var=False, store_wp=True
                            )

        # Array to store history of F-measure estimates
        self.F = np.repeat(np.nan, self.num_pts + 1)
        self.F[self.t], self._F_num, self._F_den = \
            self._calc_F(self.BB_T.theta, self.BB_P.theta, self.BB_Y.theta)

        # Covariance between t, p, y variables for each stratum (3rd dimension)
        self.cov = np.zeros([3,3,self.num_st])
        self.cov = self._calc_cov(self.BB_T.theta, self.BB_P.theta, \
                                  self.BB_Y.theta)

        # Same as F, but with a weak prior (= wp)
        self.F_wp = np.repeat(np.nan, self.num_pts + 1)
        self.F_wp[self.t], _, _ = self._calc_F(self.BB_T.theta_wp, \
                                               self.BB_P.theta_wp, \
                                               self.BB_Y.theta_wp)

        if self.record_cov_wp:
            # Same as cov, but with a weak prior (= wp)
            self.cov_wp = np.zeros([3,3,self.num_st])
            self.cov_wp = self._calc_cov(self.BB_T.theta_wp, \
                                         self.BB_P.theta_wp, self.BB_Y.theta_wp)


        self.pmf = np.zeros([self.num_st, self.num_pts + 1], dtype=float)
        self.greedy_pmf = np.zeros([self.num_st, self.num_pts + 1], dtype=float)

        # Array to store history of variance decrease
        self.decrease = np.zeros([self.num_st, self.num_pts + 1], dtype=float)
        self.decrease[:,self.t] = \
                self._calc_decrease(self.F[self.t], self.cov, self._F_num)
        self.t += 1

        for i in range(self.n_init):
            for k in self.strata.indices:
                self.sample(1, fixed_stratum = k)

    def _calc_cov(self, T_bar, P_bar, Y_bar):
        """

        """
        populations = self.strata.sizes

        cov = np.zeros([3,3,self.num_st])

        factor = populations/(populations - 1)
        cov[0,0,:] = factor * T_bar * (1 - T_bar)
        cov[1,1,:] = factor * P_bar * (1 - P_bar)
        cov[2,2,:] = factor * Y_bar * (1 - Y_bar)
        cov[0,1,:] = factor * T_bar * (1 - P_bar)
        cov[0,2,:] = factor * T_bar * (1 - Y_bar)
        cov[1,2,:] = factor * (T_bar - P_bar * Y_bar)
        cov[1,0,:] = cov[0,1,:]
        cov[2,0,:] = cov[0,2,:]
        cov[2,1,:] = cov[1,2,:]

        return cov

    def _update_BB_models(self, y, pred, stratum_idx):
        """
        Iteratively update the terms that are used to calculate the F-measure
        after a new point is sampled from stratum `stratum_idx`, at location
        `sample_loc` with label `y` and predicition `pred`.
        """
        self.BB_T.update(y * pred, stratum_idx)
        self.BB_P.update(pred, stratum_idx)
        self.BB_Y.update(y, stratum_idx)

    def _calc_F(self, T_bar, P_bar, Y_bar):
        """
        Records the latest estimate of the F-measure
        """
        t = self.t
        alpha = self.alpha
        weights = self.strata.weights

        F_num = np.dot(weights, T_bar)
        F_den = ( alpha * np.dot(weights, P_bar) +
                  (1 - alpha) * np.dot(weights, Y_bar) )

        if F_den == 0:
            return np.nan, F_num, F_den
        else:
            return F_num/F_den, F_num, F_den

    def _calc_decrease(self, F, cov, F_num):
        alpha = self.alpha
        populations = self.strata.sizes
        weights = self.strata.weights
        N = self.num_pts

        # Note adding 1 as a "pseudo-count" -- fix issue with division by zero
        # n_sampled = self.n_sampled + self.strength * weights
        n_sampled = self.n_sampled + 1
        if self.method == "var_Y":
            return (populations**2 * cov[2,2,:]) / ((n_sampled * (n_sampled + 1)) * N**2)
        elif self.method == "full":
            return ( ( cov[0,0,:] - 2 * F * (alpha * cov[0,1,:] + (1 - alpha) * cov[0,2,:])
                     + F**2 * ( (1 - alpha)**2 * cov[2,2,:] + alpha**2 * cov[1,1,:]
                                 + 2 * (1 - alpha) * alpha * cov[1,2,:] ) ) *
                   populations**2 * F**2 ) / ( (n_sampled * (n_sampled + 1)) * N**2 * F_num**2)
        else:
            raise ValueError("Unrecognised method")

    def _calc_pmf(self):
        """
        """
        t = self.t

        weights = self.strata.weights
        epsilon = self.epsilon
        decrease = self.decrease[:,t - 1]

        nonempty = (self.n_remaining > 0)
        # Among non-empty strata, sort according to "variance decrease" in decreasing order
        sorted_decrease = np.sort(decrease[nonempty])[::-1]
        # Select stratum which gives largest decrease (allowing for ties)
        best_ids = np.where((decrease == sorted_decrease[0]) & nonempty)[0]
        if self.debug:
            print("Best ids: {}".format(best_ids))
        self.greedy_pmf[best_ids,t] = 1/len(best_ids)

        # Sample uniformly from non-empty strata
        self.pmf[:,t] = weights * nonempty
        self.pmf[:,t] = self.pmf[:,t]/np.sum(self.pmf[:,t])
        self.pmf[:,t] = epsilon * self.pmf[:,t] + (1 - epsilon) * self.greedy_pmf[:,t]

    def reset(self):
        self.t = 0

        self.strata = copy.deepcopy(self._original_strata)

        self.n_remaining = copy.deepcopy(self.strata.sizes)
        self.n_sampled = np.zeros(self.num_st, dtype=int)

        self.BB_T = BetaBernoulliModel(
                            alpha_0=self._T_bar_est*self._weighted_strength,
                            beta_0=(1-self._T_bar_est)*self._weighted_strength,
                            size=None, populations=self.strata.sizes,
                            store_var=False, store_wp=True
                            )
        self.BB_P = BetaBernoulliModel(
                            alpha_0=self._P_bar_est*self._weighted_strength,
                            beta_0=(1-self._P_bar_est)*self._weighted_strength,
                            size=None, populations=self.strata.sizes,
                            store_var=False, store_wp=True
                            )
        self.BB_Y = BetaBernoulliModel(
                            alpha_0=self._Y_bar_est*self._weighted_strength,
                            beta_0=(1-self._Y_bar_est)*self._weighted_strength,
                            size=None, populations=self.strata.sizes,
                            store_var=False, store_wp=True
                            )

        self.F = np.repeat(np.nan, self.num_pts + 1)
        self.F[self.t], self._F_num, self._F_den = \
            self._calc_F(self.BB_T.theta, self.BB_P.theta, self.BB_Y.theta)

        self.cov = np.zeros([3,3,self.num_st])
        self.cov = self._calc_cov(self.BB_T.theta, self.BB_P.theta, \
                                  self.BB_Y.theta)

        self.F_wp = np.repeat(np.nan, self.num_pts + 1)
        self.F_wp[self.t], _, _ = self._calc_F(self.BB_T.theta_wp, \
                                               self.BB_P.theta_wp, \
                                               self.BB_Y.theta_wp)

        if self.record_cov_wp:
            self.cov_wp = np.zeros([3,3,self.num_st])
            self.cov_wp = self._calc_cov(self.BB_T.theta_wp, \
                                         self.BB_P.theta_wp, self.BB_Y.theta_wp)

        self.decrease = np.zeros([self.num_st, self.num_pts + 1], dtype=float)
        self.decrease[:,self.t] = \
                self._calc_decrease(self.F[self.t], self.cov, self._F_num)

        self.pmf = np.zeros([self.num_st, self.num_pts + 1], dtype=float)
        self.greedy_pmf = np.zeros([self.num_st, self.num_pts + 1], dtype=float)

        self.t += 1

        for i in range(self.n_init):
            for k in self.strata.indices:
                self.sample(1, fixed_stratum = k)

    def sample(self, n_iter, fixed_stratum = None):
        """
        Samples `n_iter` points
        fixed_stratum :

        """

        t_i = self.t
        t_f = n_iter + self.t

        # TODO Replace this assertion
        assert t_f <= self.F.shape[0]

        for t in range(t_i, t_f):
            # Check if there are any more points to sample
            if np.sum(self.n_remaining) == 0:
                print("All points have been sampled")
                return

            # Sample point
            if fixed_stratum is not None:
                stratum_idx = fixed_stratum
                sample_loc = self.strata._sample_in_stratum(stratum_idx, replace = False)
            else:
                self._calc_pmf()
                stratum_idx, sample_loc = self.strata.sample(prob_dist = self.pmf[:,self.t], replace = False)

            self.n_sampled[stratum_idx] +=1

            # Query label from oracle
            sample_idx = self.strata.allocations[stratum_idx][sample_loc]
            y = self.labels[sample_idx]
            self.strata.update_label(stratum_idx, sample_loc, y)

            # Get score
            pred = self.strata.preds[stratum_idx][sample_loc]

            self._update_BB_models(y, pred, stratum_idx)
            self.F[t], self._F_num, self._F_den = self._calc_F(self.BB_T.theta, self.BB_P.theta, self.BB_Y.theta)
            self.F_wp[t], _, _ = self._calc_F(self.BB_T.theta_wp, self.BB_P.theta_wp, self.BB_Y.theta_wp)
            self.cov = self._calc_cov(self.BB_T.theta, self.BB_P.theta, self.BB_Y.theta)
            if self.record_cov_wp:
                self.cov_wp = self._calc_cov(self.BB_T.theta_wp, self.BB_P.theta_wp, self.BB_Y.theta_wp)
            self.decrease[:,t] = self._calc_decrease(self.F[t], self.cov, self._F_num)
            self.n_remaining[stratum_idx] -= 1
            self.t += 1

            if self.debug == True:
                print("TP_term: {}, PP_term: {}, P_term: {}".format(self._TP_term, self._PP_term, self._P_term))
                print("Sampled label {} for point {} in stratum {}.".format(y, self.strata.allocations[stratum_idx][sample_loc], stratum_idx))

    def sample_until(self, n_goal):
        """
        Sample until `n_goal` labels are queried from the oracle
        """

        n_seen = self.t - 1

        if n_seen >= n_goal:
            print("Have already queried {} labels from the oracle".format(n_seen))
            return

        if n_goal > self.num_pts + 1:
            print("{} is greater than the number of points in the dataset".format(n_goal))
            return

        while n_seen < n_goal:
            self.sample(1)
            n_seen +=1
