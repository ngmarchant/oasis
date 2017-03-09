import numpy as np
from scipy.special import expit
import copy
import warnings

from .base import (BaseSampler, verify_proba)
from .stratification import (Strata, stratify_by_features, stratify_by_scores)

class BetaBernoulliModel:
    """

    """
    def __init__(self, alpha_0, beta_0, store_variance=False,
                 decaying_prior=True, store_wp=False):
        """
        Must provide size (Haldane prior) or alpha_0 and beta_0
        alpha_0 : numpy array of length `size`

        beta_0 : numpy array of length `size`

        size : integer
        """
        self.alpha_0 = copy.deepcopy(alpha_0)
        self.beta_0 = copy.deepcopy(beta_0)
        if len(alpha_0) != len(beta_0):
            raise ValueError("alpha_0 and beta_0 have inconsistent lengths")
        self.store_variance = store_variance
        self.decaying_prior = decaying_prior
        self.store_wp = store_wp
        self.size = len(alpha_0)

        # Number of positive labels sampled in each stratum (ignoring prior)
        self.alpha = np.zeros(self.size, dtype=int)
        # Number of negative labels sampled in each stratum (ignoring prior)
        self.beta = np.zeros(self.size, dtype=int)

        # Estimate of fraction of positive labels in each stratum (will
        # incorporate prior)
        self.theta = np.empty(self.size, dtype=float)
        # Estimate of variance in theta
        if self.store_variance:
            self.var_theta = np.empty(self.size, dtype=float)

        # Estimates without incorporating prior (wp = weak prior)
        if self.store_wp:
            self.theta_wp = np.empty(self.size, dtype=float)

        # Initialise
        self._calc_theta()
        if self.store_variance:
            self._calc_var_theta()

    def _calc_theta(self, wp_weight = 1e-16):
        if self.decaying_prior:
            #prior_weight = np.exp(-(self.alpha + self.beta) * self.prior_decay_const)
            #prior_weight = 1 - (self.alpha + self.beta) * self.prior_decay_const
            #prior_weight = (1 - (self.alpha + self.beta) * self.prior_decay_const)**2
            #prior_weight = 2 - np.exp((self.alpha + self.beta)**5 * np.log(2) / self.populations**5)
            #prior_weight = - expit(self.prior_decay_const * ((self.alpha + self.beta) - self.populations/2)) + 1
            n_sampled = np.clip(self.alpha + self.beta, 1, np.inf)
            prior_weight = 1/n_sampled
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
        if self.decaying_prior:
            #prior_weight = np.exp(-(self.alpha + self.beta) * self.prior_decay_const)
            #prior_weight = 1 - (self.alpha + self.beta) * self.prior_decay_const
            #prior_weight = 2 - np.exp((self.alpha + self.beta)**5 * np.log(2) / self.populations**5)
            #prior_weight = (1 - (self.alpha + self.beta) * self.prior_decay_const)**2
            #prior_weight = -expit(self.prior_decay_const * ((self.alpha + self.beta) - self.populations/2)) + 1
            #alpha = (1 - prior_weight) * self.alpha + prior_weight * self.alpha_0
            #beta = (1 - prior_weight) * self.beta + prior_weight * self.beta_0
            n_sampled = np.clip(self.alpha + self.beta, 1, np.inf)
            prior_weight = 1/n_sampled
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
        if self.store_variance:
            self._calc_var_theta()

    def reset(self):
        self.alpha = np.zeros(self.size, dtype=int)
        self.beta = np.zeros(self.size, dtype=int)
        self.theta = np.empty(self.size, dtype=float)
        if self.store_variance:
            self.var_theta = np.empty(self.size, dtype=float)
        if self.store_wp:
            self.theta_wp = np.empty(self.size, dtype=float)

        self._calc_theta()
        if self.store_variance:
            self._calc_var_theta()

class OASISSampler(BaseSampler):
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

    stratification_method

    stratification_n_bins

    stratification_n_strata

    """
    def __init__(self, alpha, oracle, predictions, scores, max_iter,
                 proba=False, prior_strength=None, epsilon=1e-3, strata=None,
                 indices = None, record_inst_hist=False, debug=False, **kwargs):
        super(OASISSampler, self).__init__(alpha, oracle, predictions,
                                           max_iter, indices, debug)
        self.scores = scores
        self.proba = verify_proba(scores, proba)
        self.prior_strength = prior_strength
        self.epsilon = epsilon
        self.strata = strata
        self.record_inst_hist = record_inst_hist
        self._requires_updating = True

        # Generate strata if not given
        if self.strata is None:
            if 'stratification_method' in kwargs:
                self.stratification_method = kwargs['stratification_method']
            else:
                self.stratification_method = 'cum_sqrt_F'
            if 'stratification_n_strata' in kwargs:
                self.stratification_n_strata = kwargs['stratification_n_strata']
            else:
                self.stratification_n_strata = 'auto'
            if 'stratification_n_bins' in kwargs:
                self.stratification_n_bins = kwargs['stratification_n_bins']
                allocations = stratify_by_scores(self.scores, \
                                        self.stratification_n_strata, \
                                        method = self.stratification_method,
                                        n_bins = self.stratification_n_bins)
            else:
                allocations = stratify_by_scores(self.scores, \
                                        self.stratification_n_strata, \
                                        method = self.stratification_method)
            self.strata = Strata(allocations)

        # Calculate mean score and mean prediction per stratum
        self.strata.mean_score = self.strata.intra_mean(self.scores)
        self.strata.mean_pred = self.strata.intra_mean(self.predictions)

        # Choose prior strength if not given
        if self.prior_strength is None:
            self.prior_strength = 2*self.strata.num_strata

        # Instantiate Beta-Bernoulli model
        gamma = self._calc_BB_prior(self.strata.mean_score,
                                      self.prior_strength, self.proba)
        self._BB_model = BetaBernoulliModel(gamma[0], gamma[1],
                                            decaying_prior=True)

        self._F_guess = self._calc_F_guess(self.alpha, self.strata.mean_pred,
                                           self._BB_model.theta,
                                           self.strata.weights)

        # Array to record history of instrumental distributions
        if self.record_inst_hist:
            self.inst_pmf = np.zeros([self.strata.num_strata, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf = np.zeros(self.strata.num_strata, dtype=float)

    def _sample_item(self):
        """
        Samples an item according to the instrumental distribution with
        replacement.
        """
        t = self.t

        # Update instrumental distribution
        self._calc_inst_pmf()

        if self.record_inst_hist:
            inst_pmf = self.inst_pmf[:,t]
        else:
            inst_pmf = self.inst_pmf

        # Sample label and record weight
        # TODO ensure that this matches method as defined in Strata
        loc, stratum_idx = self.strata.sample(pmf = inst_pmf)
        weight = self.strata.weights[stratum_idx]/inst_pmf[stratum_idx]

        return loc, weight, {'stratum': stratum_idx}

    def _update_sampler(self, ell, ell_hat, loc, weight, extra_info):
        self._BB_model.update(ell, extra_info['stratum'])

    def _calc_BB_prior(self, theta_0, prior_strength, proba):
        """
        Output
        ------
        alpha_0 : float numpy array of length K
            "alpha" hyperparameter for a sequence of K Beta-distributed rvs

        beta_0 : float numpy array of length K
            "alpha" hyperparameter for a sequence of K Beta-distributed rvs
        """
        #weighted_strength = self.weights * strength
        num_strata = len(theta_0)
        weighted_strength = prior_strength / num_strata
        if not proba:
            # Map to [0,1] interval
            theta_0 = expit(theta_0)
        alpha_0 = theta_0 * weighted_strength
        beta_0 = (1 - theta_0) * weighted_strength
        return alpha_0, beta_0

    def _calc_F_guess(self, alpha, predictions, theta, weights):
        """
        Calculates an estimate of the F-measure based on the Beta-Bernoulli
        model.
        """
        num = np.sum(theta * weights * predictions)
        den = np.sum(theta * weights * (1 - alpha) + \
                     alpha * predictions * weights)
        F_guess = 0.5 if den==0 else num/den
        return F_guess

    def _calc_inst_pmf(self):
        """

        """
        # Easy vars
        t = self.t
        epsilon = self.epsilon
        alpha = self.alpha
        predictions = self.strata.mean_pred
        weights = self.strata.weights
        p1 = self._BB_model.theta
        p0 = 1 - p1
        F = np.nan if t == 0 else self.estimate[t - 1]

        # Use an estimate for the F-measure based on the probs if it is np.nan
        if np.isnan(F) or F == 0:
            delta = 1e-10
            F = np.clip(self._F_guess, delta, 1-delta)

        # Predict positive pmf
        pp_pmf = weights * np.sqrt(alpha**2 * F**2 * p0 + (1 - F)**2 * p1)

        # Predict negative pmf
        pn_pmf = weights * (1 - alpha) * F * np.sqrt(p1)

        inst_pmf = predictions * pp_pmf + (1 - predictions) * pn_pmf
        # Normalize
        inst_pmf = inst_pmf/np.sum(inst_pmf)
        # Epsilon-greedy
        inst_pmf = epsilon * weights + (1 - epsilon) * inst_pmf

        if self.record_inst_hist:
            self.inst_pmf[:,t] = inst_pmf
        else:
            self.inst_pmf = inst_pmf

    def reset(self):
        """
        Resets the instance to begin sampling again
        """
        super(OASISSampler, self).reset()
        self.strata.reset()
        self._BB_model.reset()

        # Array to record history of instrumental distributions
        if self.record_inst_hist:
            self.inst_pmf = np.zeros([self.strata.num_strata, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf = np.zeros(self.strata.num_strata, dtype=float)
