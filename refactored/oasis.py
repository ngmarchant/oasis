import numpy as np
import warnings
import sys


class BetaBernoulliModel:
    """Beta-Bernoulli model for the stratified oracle probabilities

    Parameters
    ----------
    alpha_0 : numpy.ndarray, shape=(n_strata,)
        alpha prior hyperparameter

    beta_0 : numpy.ndarray, shape=(n_strata,)
        beta prior hyperparameter

    store_variance : bool, optional, default False
        whether to store an estimate of the variance of theta

    decaying_prior : bool, optional, default True
        whether to make the prior strength decay as 1/n_k, where n_k is the
        number of items sampled from stratum k at the current iteration.

    store_weak_prior : bool, optional, default False
        whether to store estimates based on a very weak prior.

    Attributes
    ----------
    alpha_ : numpy.ndarray, shape=(n_strata,)
        posterior value of alpha (excluding prior)

    beta_ : numpy.ndarray, shape=(n_strata,)
        posterior value of beta (excluding prior)

    theta_ : numpy.ndarray, shape=(n_strata,)
        posterior estimate of theta

    var_theta_ : numpy.ndarray, shape=(n_strata,)
        posterior estimate of var(theta)
    """
    def __init__(self, alpha_0, beta_0, store_variance=False,
                 decaying_prior=True, store_wp=False):

        if len(alpha_0) != len(beta_0):
            raise ValueError("alpha_0 and beta_0 have inconsistent lengths")

        self.alpha_0 = alpha_0.copy()
        self.beta_0 = beta_0.copy()

        self.store_variance = store_variance
        self.decaying_prior = decaying_prior
        self.store_wp = store_wp
        self._size = len(alpha_0)

        # Number of "1" and "0" label resp. (excluding prior)
        self.alpha_ = np.zeros(self._size, dtype=int)
        self.beta_ = np.zeros(self._size, dtype=int)

        # Estimate of fraction of positive labels in each stratum
        self.theta_ = np.empty(self._size, dtype=float)
        # Estimate of variance in theta
        if self.store_variance:
            self.var_theta_ = np.empty(self._size, dtype=float)

        # Estimates without incorporating prior (wp = weak prior)
        if self.store_wp:
            self.theta_wp_ = np.empty(self._size, dtype=float)
            self._wp_weight = 1e-20

        # Initialise estimates
        self._calc_theta()
        if self.store_variance:
            self._calc_var_theta()

    def calc_theta(self):
        """Calculate an estimate of theta"""
        if self.decaying_prior:
            n_sampled = np.clip(self.alpha_ + self.beta_, 1, np.inf)
            prior_weight = 1/n_sampled
            alpha = self.alpha_ + prior_weight * self.alpha_0
            beta = self.beta_ + prior_weight * self.beta_0
        else:
            alpha = self.alpha_ + self.alpha_0
            beta = self.beta_ + self.beta_0

        # Mean of Beta-distributed rv
        self.theta_ = alpha / (alpha + beta)

        # NEW: calculate theta assuming weak prior
        if self.store_wp:
            alpha = self.alpha_ + self._wp_weight * self.alpha_0
            beta = self.beta_ + self._wp_weight * self.beta_0
            self.theta_wp_ = alpha / (alpha + beta)

    def calc_var_theta(self):
        """Calculate an estimate of the var(theta)"""
        if self.decaying_prior:
            n_sampled = np.clip(self.alpha_ + self.beta_, 1, np.inf)
            prior_weight = 1/n_sampled
            alpha = self.alpha_ + prior_weight * self.alpha_0
            beta = self.beta_ + prior_weight * self.beta_0
        else:
            alpha = self.alpha_ + self.alpha_0
            beta = self.beta_ + self.beta_0
        # Variance of Beta-distributed rv
        self.var_theta_ = alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))

    def update(self, ell, k):
        """Update the posterior and estimates after a label is sampled

        Parameters
        ----------
        ell : int
            sampled label: 0 or 1

        k : int
            index of stratum where label was sampled
        """
        self.alpha_[k] += ell
        self.beta_[k] += 1 - ell

        self._calc_theta()
        if self.store_variance:
            self._calc_var_theta()

    def reset(self):
        """Reset the instance to its initial state"""
        self.alpha_ = np.zeros(self._size, dtype=int)
        self.beta_ = np.zeros(self._size, dtype=int)
        self.theta_ = np.empty(self._size, dtype=float)
        if self.store_variance:
            self.var_theta_ = np.empty(self._size, dtype=float)
        if self.store_wp:
            self.theta_wp_ = np.empty(self._size, dtype=float)

        self.calc_theta()
        if self.store_variance:
            self.calc_var_theta()

    @classmethod
    def from_prior(cls, theta_0, prior_strength, **kwargs):
        """Generate a prior for the BB model

        Parameters
        ----------
        theta_0 : array-like, shape=(n_strata,)
            array of oracle probabilities (probability of a "1" label)
            for each stratum. This is just a guess.

        Returns
        -------
        alpha_0 : numpy.ndarray, shape=(n_strata,)
            "alpha" hyperparameters for an ensemble of Beta-distributed rvs

        beta_0 : numpy.ndarray, shape=(n_strata,)
            "beta" hyperparameters for an ensemble of Beta-distributed rvs
        """
        #: Easy vars
        # weighted_strength = self.weights * strength
        n_strata = len(theta_0)
        weighted_strength = prior_strength / n_strata
        alpha_0 = theta_0 * weighted_strength
        beta_0 = (1 - theta_0) * weighted_strength
        return cls(alpha_0, beta_0, **kwargs)



class OASISSampler:
    """Optimal Asymptotic Sequential Importance Sampling (OASIS) for estimation
    of the weighted F-measure.

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an adaptive instrumental
    distribution that minimises asymptotic variance. See reference
    [Marchant2017]_ for details.

    Parameters
    ----------
    alpha : float
        Weight for the F-measure. Valid weights are on the interval [0, 1].
        ``alpha == 1`` corresponds to precision, ``alpha == 0`` corresponds to
        recall, and ``alpha == 0.5`` corresponds to the balanced F-measure.

    predictions : array-like, shape=(n_items,n_class)
        Predicted labels for the items in the pool. Rows represent items and
        columns represent different classifiers under evaluation (i.e. more
        than one classifier may be evaluated in parallel). Valid labels are 0
        or 1.

    scores : array-like, shape=(n_items,n_class)
        Scores which quantify the confidence in the classifiers' predictions.
        Rows represent items and columns represent different classifiers under
        evaluation. High scores indicate a high confidence that the true label
        is 1 (and vice versa for label 0). It is recommended that the scores
        be scaled to the interval [0,1]. If the scores lie outside [0,1] they
        will be automatically re-scaled by applying the logisitic function.

    oracle : function
        Function that returns ground truth labels for items in the pool. The
        function should take an item identifier as input (i.e. its
        corresponding row index) and return the ground truth label. Valid
        labels are 0 or 1.

    proba : array-like, dtype=bool, shape=(n_class,), optional, default None
        Indicates whether the scores are probabilistic, i.e. on the interval
        [0, 1] for each classifier under evaluation. If proba is False for
        a classifier, then the corresponding scores will be re-scaled by
        applying the logistic function. If None, proba will default to
        False for all classifiers.

    epsilon : float, optional, default 1e-3
        Epsilon-greedy parameter. Valid values are on the interval [0, 1]. The
        "asymptotically optimal" distribution is sampled from with probability
        `1 - epsilon` and the passive distribution is sampled from with
        probability `epsilon`. The sampling is close to "optimal" for small
        epsilon.

    prior_strength : float, optional, default None
        Quantifies the strength of the prior. May be interpreted as the number
        of pseudo-observations.

    max_iter : int, optional, default None
        Maximum number of iterations to expect for pre-allocating arrays.
        Once this limit is reached, sampling can no longer continue. If no
        value is given, defaults to n_items.

    strata : Strata instance, optional, default None
        Describes how to stratify the pool. If not given, the stratification
        will be done automatically based on the scores given. Additional
        keyword arguments may be passed to control this automatic
        stratification (see below).

    Other Parameters
    ----------------
    opt_class : array-like, dtype=bool, shape=(n_class,), optional, default None
        Indicates which classifiers to use in calculating the optimal
        distribution (and prior and strata). If opt_class is False for a
        classifier, then its predictions and scores will not be used in
        calculating the optimal distribution, however estimates of its
        performance will still be calculated.

    decaying_prior : bool, optional, default True
        Whether to make the prior strength decay as 1/n_k, where n_k is the
        number of items sampled from stratum k at the current iteration. This
        is a greedy strategy which may yield faster convergence of the estimate.

    record_inst_hist : bool, optional, default False
        Whether to store the instrumental distribution used at each iteration.
        This requires extra memory, but can be useful for assessing
        convergence.

    identifiers : array-like, optional, default None
        Unique identifiers for the items in the pool. Must match the row order
        of the "predictions" parameter. If no value is given, defaults to
        [0, 1, ..., n_items].

    debug : bool, optional, default False
        Whether to print out verbose debugging information.

    **kwargs :
        Optional keyword arguments. Includes 'stratification_method',
        'stratification_n_strata', and 'stratification_n_bins'.

    Attributes
    ----------
    estimate_ : numpy.ndarray
        F-measure estimates for each iteration.

    queried_oracle_ : numpy.ndarray
        Records whether the oracle was queried at each iteration (True) or
        whether a cached label was used (False).

    cached_labels_ : numpy.ndarray, shape=(n_items,)
        Previously sampled ground truth labels for the items in the pool. Items
        which have not had their labels queried are recorded as NaNs. The order
        of the items matches the row order for the "predictions" parameter.

    t_ : int
        Iteration index.

    inst_pmf_ : numpy.ndarray, shape=(n_strata,) or (n_strata, max_iter)
        Epsilon-greedy instrumental pmf used for sampling. If
        ``record_inst_hist == False`` only the most recent pmf is returned,
        otherwise returns the entire history of pmfs in a 2D array.

    References
    ----------
    .. [Marchant2017] N. G. Marchant and B. I. P. Rubinstein, In Search of an
       Entity Resolution OASIS: Optimal Asymptotic Sequential Importance
       Sampling, arXiv:1703.00617 [cs.LG], Mar 2017.
    """
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 epsilon=1e-3, opt_class=None, prior_strength=None,
                 decaying_prior=True, strata=None, record_inst_hist=False,
                 max_iter=None, identifiers=None, debug=False, **kwargs):
        super(OASISSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, identifiers, True, debug)

        # scores

        self.scores = scores

        # Calculate mean prediction per stratum
        self._preds_avg_in_strata = self.strata.intra_mean(self.predictions)

        # Choose prior strength if not given
        self.prior_strength = prior_strength or 2*self.strata.n_strata_


        # Instantiate Beta-Bernoulli model using probabilities averaged over
        # opt_class
        theta_0 = self.strata.intra_mean(self.scores)
        self._BB_model = BetaBernoulliModel.from_prior(theta_0, self.prior_strength, decaying_prior=True)
        self._F_guess = self._calc_F_guess(self.alpha,
                                           self._preds_avg_in_strata,
                                           self._BB_model.theta_,
                                           self.strata.weights_)

        # Array to record history of instrumental distributions

        self._inst_pmf = np.zeros(self.strata.n_strata_, dtype=float)

    # @property
    # def inst_pmf_(self):
    #     if self.record_inst_hist:
    #         return self._inst_pmf[:, 0:self.t_]
    #     else:
    #         return self._inst_pmf
    # @inst_pmf_.setter
    # def inst_pmf_(self, value):
    #     raise AttributeError("can't set attribute.")
    # @inst_pmf_.deleter
    # def inst_pmf_(self):
    #     raise AttributeError("can't delete attribute.")

    def _sample_item(self, **kwargs):
        """Sample an item from the pool according to the instrumental
        distribution
        """
        t = self.t_

        # Update instrumental distribution
        self._calc_inst_pmf()

        if self.record_inst_hist:
            inst_pmf = self._inst_pmf[:,t]
        else:
            inst_pmf = self._inst_pmf

        # Sample label and record weight
        loc, stratum_idx = self.strata.sample(pmf = inst_pmf)
        weight = self.strata.weights_[stratum_idx]/inst_pmf[stratum_idx]

        return loc, weight, {'stratum': stratum_idx}

    def _update_estimate_and_sampler(self, ell, ell_hat, weight, extra_info,
                                     **kwargs):
        #: Updating the estimate is handled in the base class
        super(OASISSampler, self)._update_estimate_and_sampler(ell, ell_hat, \
                                                weight, extra_info, **kwargs)

        #: Update the instrumental distribution by updating the BB model
        self._BB_model.update(ell, extra_info['stratum'])

    # def _calc_BB_prior(self, theta_0):
    #     """Generate a prior for the BB model
    #
    #     Parameters
    #     ----------
    #     theta_0 : array-like, shape=(n_strata,)
    #         array of oracle probabilities (probability of a "1" label)
    #         for each stratum. This is just a guess.
    #
    #     Returns
    #     -------
    #     alpha_0 : numpy.ndarray, shape=(n_strata,)
    #         "alpha" hyperparameters for an ensemble of Beta-distributed rvs
    #
    #     beta_0 : numpy.ndarray, shape=(n_strata,)
    #         "beta" hyperparameters for an ensemble of Beta-distributed rvs
    #     """
    #     #: Easy vars
    #     prior_strength = self.prior_strength
    #
    #     #weighted_strength = self.weights * strength
    #     n_strata = len(theta_0)
    #     weighted_strength = prior_strength / n_strata
    #     alpha_0 = theta_0 * weighted_strength
    #     beta_0 = (1 - theta_0) * weighted_strength
    #     return alpha_0, beta_0

    def _calc_F_guess(self, alpha, predictions, theta, weights):
        """Calculate an estimate of the F-measure based on the scores"""
        num = np.sum(predictions.T * theta * weights, axis=1)
        den = np.sum((1 - alpha) * theta * weights + \
                     alpha * predictions.T * weights, axis=1)
        F_guess = num/den
        # Ensure guess is not undefined
        F_guess[den==0] = 0.5
        return F_guess

    def _calc_inst_pmf(self):
        """Calculate the epsilon-greedy instrumental distribution"""
        # Easy vars
        t = self.t_
        epsilon = self.epsilon
        alpha = self.alpha
        preds = self._preds_avg_in_strata
        weights = self.strata.weights_[:,np.newaxis]
        p1 = self._BB_model.theta_
        p0 = 1 - p1
        if t==0:
            F = self._F_guess[self.opt_class]
        else:
            F = self._estimate[t - 1, self.opt_class]
            # Fill in non-finite estimates with the initial guess
            nonfinite = ~np.isfinite(F)
            F[nonfinite] = self._F_guess[self.opt_class][nonfinite]

        # Calculate optimal instrumental pmf
        sqrt_arg = np.sum(preds * (alpha**2 * F**2 * p0 + (1 - F)**2 * p1) + \
                          (1 - preds) * (1 - alpha)**2 * F**2 * p1, \
                          axis=1, keepdims=True) #: sum is over classifiers
        inst_pmf = weights * np.sqrt(sqrt_arg)
        # Normalize
        inst_pmf /= np.sum(inst_pmf)
        # Epsilon-greedy: (1 - epsilon) q + epsilon * p
        inst_pmf *= (1 - epsilon)
        inst_pmf += epsilon * weights
        self._inst_pmf = inst_pmf



if __name__ == '__main__':
    from oasis.experiments import Data
    data = Data()
    data.read_h5('Amazon-GoogleProducts-test.h5')

    def oracle(idx):
        return data.labels[idx]
    alpha = 0.5

    smplr =OASISSampler(alpha, data.preds, data.scores, oracle)
    smplr.sample_distinct(5000)  #: query labels for 5000 distinct items

