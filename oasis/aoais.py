import numpy as np
from scipy.special import expit
import copy
import warnings

class BetaBernoulliModel:
    """

    """
    def __init__(self, alpha_0=None, beta_0=None, size = None,
                 populations=None, store_var=False, store_wp=False):
        """
        Must provide size (Haldane prior) or alpha_0 and beta_0
        alpha_0 : numpy array of length `size`

        beta_0 : numpy array of length `size`

        size : integer
        """
        if size is None and alpha_0 is None and beta_0 is None:
            raise Exception("No valid arguments given. Must provide alpha_0 \
                             and beta_0, or size of model.")
        if size is None:
            # Ignore given size
            self.size = len(alpha_0)
        else:
            self.size = size
        if alpha_0 is None or beta_0 is None:
            # Approximate Haldane prior
            self.alpha_0 = np.repeat(np.finfo(float).eps, self.size)
            self.beta_0 = np.repeat(np.finfo(float).eps, self.size)
        else:
            # Use prior given
            self.alpha_0 = copy.deepcopy(alpha_0)
            self.beta_0 = copy.deepcopy(beta_0)
            if len(alpha_0) != len(beta_0):
                raise ValueError("alpha_0 and beta_0 have inconsistent lengths")
        self.populations = populations
        self.store_var = store_var
        self.store_wp = store_wp

        # Number of positive labels sampled in each stratum (ignoring prior)
        self.alpha = np.zeros(self.size, dtype=int)
        # Number of negative labels sampled in each stratum (ignoring prior)
        self.beta = np.zeros(self.size, dtype=int)

        # Estimate of fraction of positive labels in each stratum (will
        # incorporate prior)
        self.theta = np.empty(self.size, dtype=float)
        # Estimate of variance in theta
        if self.store_var:
            self.var_theta = np.empty(self.size, dtype=float)

        # Estimates without incorporating prior (wp = weak prior)
        if self.store_wp:
            self.theta_wp = np.empty(self.size, dtype=float)

        # Initialise
        self._calc_theta()
        if self.store_var:
            self._calc_var_theta()

    def _calc_theta(self, wp_weight = 1e-16):
        if self.populations is not None:
            #prior_weight = np.exp(-(self.alpha + self.beta) * self.prior_decay_const)
            #prior_weight = 1 - (self.alpha + self.beta) * self.prior_decay_const
            #prior_weight = (1 - (self.alpha + self.beta) * self.prior_decay_const)**2
            # prior_weight = 2 - np.exp((self.alpha + self.beta)**5 * np.log(2) / self.populations**5)
            n_sampled = np.clip(self.alpha + self.beta, a_min = 1, a_max = np.inf)
            prior_weight = 1/n_sampled
            #prior_weight = - expit(self.prior_decay_const * ((self.alpha + self.beta) - self.populations/2)) + 1
            alpha = self.alpha + prior_weight * self.alpha_0
            beta = self.beta + prior_weight * self.beta_0
        else:
            alpha = self.alpha + self.alpha_0
            beta = self.beta + self.beta_0
        # Mean of Beta-distributed rv
        self.theta = alpha / (alpha + beta)

        # NEW: calculate theta assuming weak prior
        if self.store_wp:
            alpha = self.alpha + wp_weight * self.alpha_0
            beta = self.beta + wp_weight * self.beta_0
            self.theta_wp = alpha / (alpha + beta)

    def _calc_var_theta(self):
        if self.populations is not None:
            #prior_weight = np.exp(-(self.alpha + self.beta) * self.prior_decay_const)
            #prior_weight = 1 - (self.alpha + self.beta) * self.prior_decay_const
            #prior_weight = 2 - np.exp((self.alpha + self.beta)**5 * np.log(2) / self.populations**5)
            n_sampled = np.clip(self.alpha + self.beta, a_min = 1, a_max = np.inf)
            prior_weight = 1/n_sampled
            #prior_weight = (1 - (self.alpha + self.beta) * self.prior_decay_const)**2
            #prior_weight = -expit(self.prior_decay_const * ((self.alpha + self.beta) - self.populations/2)) + 1
            #alpha = (1 - prior_weight) * self.alpha + prior_weight * self.alpha_0
            #beta = (1 - prior_weight) * self.beta + prior_weight * self.beta_0
            alpha = self.alpha + prior_weight * self.alpha_0
            beta = self.beta + prior_weight * self.beta_0
        else:
            alpha = self.alpha + self.alpha_0
            beta = self.beta + self.beta_0
        # Variance of Beta-distributed rv
        self.var_theta = alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))

    def update(self, y, k):
        """
        Updates the Beta-Bernoulli model given a point sampled from stratum `k`
        with label `y`.
        """
        self.alpha[k] = self.alpha[k] + y
        self.beta[k] = self.beta[k] + 1 - y

        self._calc_theta()
        if self.store_var:
            self._calc_var_theta()

    def split(self, k_splt, pop_splt, pop_new):
        """
        k_splt : integer
            stratum index to split

        pop_splt : integer
            population of stratum `k_splt` after the split

        pop_new : integer
            population of newly-created stratum after the split
        """
        # Increase length of arrays by 1
        self.alpha = np.append(self.alpha, 0)
        self.beta = np.append(self.beta, 0)
        self.alpha_0 = np.append(self.alpha_0, 0)
        self.beta_0 = np.append(self.beta_0, 0)
        self.theta = np.append(self.theta, 0)
        if self.store_var:
            self.var_theta = np.append(self.var_theta, 0)
        self.size += 1

        k_new = self.size - 1
        pop_tot = pop_splt + pop_new

        # Redistribute prior proportionately
        self.alpha_0[k_new] = self.alpha_0[k_splt] * pop_new/pop_tot
        self.alpha_0[k_splt] = self.alpha_0[k_splt] * pop_splt/pop_tot
        self.beta_0[k_new] = self.beta_0[k_splt] * pop_new/pop_tot
        self.beta_0[k_splt] = self.beta_0[k_splt] * pop_splt/pop_tot

        self._calc_theta()
        self._calc_var_theta()


class AOAIS:
    # TODO Fix the documentation
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

    max_iter : int, optional, default None
        maximum number of iterations permitted (used to pre-allocate arrays for
        storing the sampling history). Defaults to the number of data points.

    prior_strength : float, optional, default to number of strata
            strength of prior -- can be interpreted as the number of pseudo-
            observations

    epsilon : float, optional, default 1e-3
        epsilon-greedy parameter. Takes a value on the closed unit interval.
        The "optimal" distribution is used with probability `1 - epsilon`, and
        the passive distribution is used with probability `epsilon`. The
        sampling is close to "optimal" for small epsilon.

    debug : bool, optional, default False
        if True, prints debugging information.

    """
    def __init__(self, labels, strata, alpha = 0.5, epsilon = 1e-3,
                   prior_strength = None, max_iter = None, pmf_history = False,
                   debug = False):
        self.labels = labels
        self._original_strata = strata
        self.strata = copy.deepcopy(strata)
        self.alpha = alpha
        self.epsilon = epsilon
        self.prior_strength = prior_strength
        self._max_iter = max_iter
        self._F_prob_est = None
        self.pmf_history = pmf_history
        self.debug = debug

        self.t = 0

        if self._max_iter is None:
            self._max_iter = self.strata.num_pts

        # Terms used to calculate F-measure (we update them iteratively)
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0

        # Instantiate Beta-Bernoulli model
        self.BB_model = \
                BetaBernoulliModel(*self._calc_prior(), populations = self.strata.populations)

        # Array to record history of F-measure estimates
        self.F = np.repeat(np.nan, self._max_iter)

        # Array to record history of instrumental distributions
        if self.pmf_history:
            self.pmf = np.zeros([self.strata.num_st, self._max_iter], dtype=float)
        else:
            self.pmf = np.zeros(self.strata.num_st, dtype=float)

        # Array to record whether oracle was queried at each iteration
        self.queried_oracle = np.repeat(False, self._max_iter)

    def _calc_prior(self):
        """
        Output
        ------
        alpha_0 : float numpy array of length K
            "alpha" hyperparameter for a sequence of K Beta-distributed rvs

        beta_0 : float numpy array of length K
            "alpha" hyperparameter for a sequence of K Beta-distributed rvs
        """
        prior_strength = self.prior_strength
        if prior_strength is None:
            prior_strength = self.strata.num_st

        #weighted_strength = self.weights * strength
        weighted_strength = prior_strength / self.strata.num_st

        if self.strata.calibrated_score:
            theta_0 = self.strata.mean_score
        else:
            theta_0 = expit(self.strata.mean_score)

        alpha_0 = theta_0 * weighted_strength
        beta_0 = (1 - theta_0) * weighted_strength

        return alpha_0, beta_0

    def _update_F_terms(self, y, yhat, w):
        """
        Iteratively update the terms that are used to calculate the F-measure
        after a new point is sampled with weight `w`, label `y` and prediction
        `yhat`.
        """
        if y == 1 and yhat == 1:
            # Point is true positive
            self._TP_term = self._TP_term + w
            self._PP_term = self._PP_term + w
            self._P_term = self._P_term + w
        elif yhat == 1:
            # Point is false positive
            self._PP_term = self._PP_term + w
        elif y == 1:
            # Point is false negative
            self._P_term = self._P_term + w

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

    def _calc_F_prob_est(self):
        """
        Calculates and estimate of the F-measure based on the Beta-Bernoulli
        model and stores the result in self._F_prob_est
        """

        theta = self.BB_model.theta
        weights = self.strata.weights
        alpha = self.alpha
        preds = self.strata.mean_pred
        F_num = np.sum(theta * weights * preds)
        F_den = np.sum(theta * weights * (1 - alpha) + \
                       alpha * preds * weights)
        self._F_prob_est = F_num/F_den

    def _calc_optimal_dist(self):
        """

        """
        # Easy vars
        epsilon = self.epsilon
        alpha = self.alpha
        preds = self.strata.mean_pred
        weights = self.strata.weights
        t = self.t

        # Use most recent estimates of F and theta
        F_est = np.nan if t == 0 else self.F[t - 1]
        p1 = self.BB_model.theta
        p0 = 1 - p1

        # Use an estimate for the F-measure based on the probs if it is np.nan
        if np.isnan(F_est) or F_est == 0:
            if self._F_prob_est is None:
                self._calc_F_prob_est()
            if self._F_prob_est < 1e-10:
                F_est = 1e-10
            else:
                F_est = self._F_prob_est

        # Predict positive pmf
        pp_pmf = ( weights * np.sqrt((alpha**2 *
                                    F_est**2 * p0 + (1 - F_est)**2 * p1)) )

        # Predict negative pmf
        pn_pmf = ( weights * (1 - alpha) * F_est * np.sqrt(p1) )

        pmf = preds * pp_pmf + (1 - preds) * pn_pmf
        # Normalize
        pmf = pmf/np.sum(pmf)
        # Weight by passive pmf
        pmf = epsilon * weights + (1 - epsilon) * pmf

        if self.pmf_history:
            self.pmf[:,t] = pmf
        else:
            self.pmf = pmf

    def reset(self):
        """
        Resets the instance to begin sampling again
        """
        self.t = 0

        self.strata = copy.deepcopy(self._original_strata)

        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0

        self.BB_model = \
                BetaBernoulliModel(*self._calc_prior(), populations = self.strata.populations)

        self.F = np.repeat(np.nan, self._max_iter)

        # Array to record history of instrumental distributions
        if self.pmf_history:
            self.pmf = np.zeros([self.strata.num_st, self._max_iter], dtype=float)
        else:
            self.pmf = np.zeros(self.strata.num_st, dtype=float)
        self.queried_oracle = np.repeat(False, self._max_iter)

    def sample(self, n_iter):
        """
        Samples `n_iter` points
        """

        t_i = self.t
        t_f = n_iter + self.t

        assert t_f <= self.F.shape[0]

        for t in range(t_i, t_f):
            # Calculate pmf
            self._calc_optimal_dist()

            if self.pmf_history:
                pmf = self.pmf[:,t]
            else:
                pmf = self.pmf

            # Sample label and record weight
            stratum_idx, sample_loc = self.strata.sample(prob_dist = pmf)
            w = self.strata.weights[stratum_idx]/pmf[stratum_idx]

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
                print("Sampled label {} for point {} in stratum {}. Weight is {}.".format(y,self.strata.allocations[stratum_idx][sample_loc], stratum_idx, w))


            if ( self.strata.splitting and
                 ( ( y == 1 and self.BB_model.alpha[stratum_idx] == 0 and
                     self.BB_model.beta[stratum_idx] > 0 ) or
                   ( y == 0 and self.BB_model.beta[stratum_idx] == 0 and
                     self.BB_model.alpha[stratum_idx] > 0 ) ) and
                   self.strata.populations[stratum_idx] > 1 ):
                # Split if the sampled label is different from those previously
                # seen in the stratum
                if self.debug:
                    print("Splitting stratum {}".format(stratum_idx))

                # Split stratum
                self.strata.split(stratum_idx, sample_loc, y)

                # Get populations of split stratum and new stratum
                pop_splt = self.strata.populations[stratum_idx]
                pop_new = self.strata.populations[-1]

                # Fix BB model
                self.BB_model.split(stratum_idx, pop_splt, pop_new)

                # Change stratum_idx so that new stratum will be updated below
                stratum_idx = self.strata.indices[-1]

                # Increase size of pmf array
                self.pmf = np.append(self.pmf, 0)

            self.BB_model.update(y, stratum_idx)
            self._update_F_terms(y, pred, w)
            self._update_F()

            self.t = self.t + 1

    def sample_until(self, n_goal):
        """
        Sample until `n_goal` labels are queried from the oracle
        """

        n_seen = np.sum(self.queried_oracle)

        if n_seen >= n_goal:
            print("Have already queried {} labels from the oracle".format(n_seen))
            return

        if n_goal > self._max_iter:
            print("{} is greater than max_iter = {}".format(n_goal,self._max_iter))
            return

        while n_seen < n_goal:
            self.sample(1)
            n_seen = n_seen + self.queried_oracle[self.t - 1]*1
