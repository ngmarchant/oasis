import numpy as np
from scipy.special import expit
import copy
import warnings

from .base import (BaseSampler, verify_proba)
from .stratification import (Strata, stratify_by_features, stratify_by_scores,
                             auto_stratify)

class BetaBernoulliModel:
    """Beta-Bernoulli model for the stratified oracle probabilities

    Parameters
    ----------
    alpha_0 : numpy.ndarray, shape=(num_strata,)
        alpha prior hyperparameter

    beta_0 : numpy.ndarray, shape=(num_strata,)
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
    alpha_ : numpy.ndarray, shape=(num_strata,)
        posterior value of alpha (excluding prior)

    beta_ : numpy.ndarray, shape=(num_strata,)
        posterior value of beta (excluding prior)

    theta_ : numpy.ndarray, shape=(num_strata,)
        posterior estimate of theta

    var_theta_ : numpy.ndarray, shape=(num_strata,)
        posterior estimate of var(theta)
    """
    def __init__(self, alpha_0, beta_0, store_variance=False,
                 decaying_prior=True, store_wp=False):
        self.alpha_0 = copy.deepcopy(alpha_0)
        self.beta_0 = copy.deepcopy(beta_0)
        if len(alpha_0) != len(beta_0):
            raise ValueError("alpha_0 and beta_0 have inconsistent lengths")
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
            self.theta_wp = np.empty(self._size, dtype=float)
            self._wp_weight = 1e-16

        # Initialise estimates
        self._calc_theta()
        if self.store_variance:
            self._calc_var_theta()

    def _calc_theta(self):
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
            self.theta_wp = alpha / (alpha + beta)

    def _calc_var_theta(self):
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
        self.var_theta_ = ( alpha * beta /
                            ((alpha + beta)**2 * (alpha + beta + 1)) )

    def update(self, ell, k):
        """Update the posterior and estimates after a label is sampled

        Parameters
        ----------
        ell : int
            sampled label: 0 or 1

        k : int
            index of stratum where label was sampled
        """
        self.alpha_[k] = self.alpha_[k] + ell
        self.beta_[k] = self.beta_[k] + 1 - ell

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
            self.theta_wp = np.empty(self._size, dtype=float)

        self._calc_theta()
        if self.store_variance:
            self._calc_var_theta()

class OASISSampler(BaseSampler):
    """Adaptive importance sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an adaptive instrumental
    distribution that minimises asymptotic variance. See reference
    [Marchant2017]_ for details.

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

    prior_strength : float, optional, default None
        parameter which quantifies the strength of the prior. May be
        interpreted as the number of pseudo-observations.

    max_iter : int, optional, default None
        space for storing the sampling history is limited to a maximum
        number of iterations. Once this limit is reached, sampling can no
        longer continue. If no value is given, defaults to the size of
        the pool.

    strata : Strata instance, optional, default None
        describes how to stratify the pool. If not given, the stratification
        will be done automatically based on the scores given. Additional
        keyword arguments may be passed to control this automatic
        stratification (see below).

    Other Parameters
    ----------------
    decaying_prior : bool, optional, default True
        whether to make the prior strength decay as 1/n_k, where n_k is the
        number of items sampled from stratum k at the current iteration. This
        is a greedy strategy which may yield faster convergence of the estimate.

    record_inst_hist : bool, optional, default False
        whether to store the instrumental distribution used at each iteration.
        This requires extra memory, but can be useful for assessing
        convergence.

    indices : array-like, optional, default None
        ordered array of unique identifiers for the items in the pool.
        Should match the order of the "predictions" parameter. If no value is
        given, defaults to [0, 1, ..., pool_size].

    debug : bool, optional, default False
        whether to print out verbose debugging information.

    **kwargs :
        optional keyword arguments (see section below).

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

    inst_pmf_ : numpy.ndarray, shape=(num_strata,) or (num_strata, max_iter)
        epsilon-greedy instrumental pmf used for sampling. If
        ``record_inst_hist == False`` only the most recent pmf is returned,
        else returns the entire history of pmfs in a 2D array.

    Optional **kwargs
    -----------------
    stratification_method : {'cum_sqrt_F' or 'equal_size'}
        stratification method to use. See TODO

    stratification_n_bins : int
        number of bins to use when approximating the score distribution. See
        TODO

    stratification_n_strata : int
        goal number of strata (not guaranteed). See TODO

    .. [Marchant2017] N. G. Marchant and B. I. P. Rubinstein, In Search of an
    Entity Resolution OASIS: Optimal Asymptotic Sequential Importance Sampling,
    arXiv:1703.00617 [cs.LG], Mar 2017.
    """
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 epsilon=1e-3, prior_strength=None, decaying_prior=True,
                 strata=None, record_inst_hist=False, max_iter=None,
                 indices = None, debug=False, **kwargs):
        super(OASISSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, indices, debug)
        self.scores = scores
        self.proba = verify_proba(scores, proba)
        self.prior_strength = prior_strength
        self.epsilon = epsilon
        self.strata = strata
        self.record_inst_hist = record_inst_hist
        self.decaying_prior = decaying_prior
        self._requires_updating = True

        # Generate strata if not given
        if self.strata is None:
            self.strata = auto_stratify(self.scores, **kwargs)

        # Calculate mean score and mean prediction per stratum
        self.strata.mean_score = self.strata.intra_mean(self.scores)
        self.strata.mean_pred = self.strata.intra_mean(self.predictions)

        # Choose prior strength if not given
        if self.prior_strength is None:
            self.prior_strength = 2*self.strata.num_strata_

        # Instantiate Beta-Bernoulli model
        gamma = self._calc_BB_prior(self.strata.mean_score,
                                      self.prior_strength, self.proba)
        self._BB_model = BetaBernoulliModel(gamma[0], gamma[1],
                                            decaying_prior=self.decaying_prior)

        self._F_guess = self._calc_F_guess(self.alpha, self.strata.mean_pred,
                                           self._BB_model.theta_,
                                           self.strata.weights_)

        # Array to record history of instrumental distributions
        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.num_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.num_strata_, dtype=float)

    def _sample_item(self):
        """Sample an item from the pool according to the instrumental
        distribution
        """
        t = self.t_

        # Update instrumental distribution
        self._calc_inst_pmf()

        if self.record_inst_hist:
            inst_pmf = self.inst_pmf_[:,t]
        else:
            inst_pmf = self.inst_pmf_

        # Sample label and record weight
        loc, stratum_idx = self.strata.sample(pmf = inst_pmf)
        weight = self.strata.weights_[stratum_idx]/inst_pmf[stratum_idx]

        return loc, weight, {'stratum': stratum_idx}

    def _update_sampler(self, ell, ell_hat, loc, weight, extra_info):
        """Update the instrumental distribution by updating the BB model"""
        self._BB_model.update(ell, extra_info['stratum'])

    def _calc_BB_prior(self, theta_0):
        """Generate a prior for the BB model

        Parameters
        ----------
        theta_0 : array-like, shape=(num_strata,)
            array of oracle probabilities (probability of a "1" label)
            for each stratum. This is just a guess.

        Returns
        -------
        alpha_0 : numpy.ndarray, shape=(num_strata,)
            "alpha" hyperparameters for an ensemble of Beta-distributed rvs

        beta_0 : numpy.ndarray, shape=(num_strata,)
            "beta" hyperparameters for an ensemble of Beta-distributed rvs
        """
        #: Easy vars
        prior_strength = self.prior_strength
        proba = self.proba

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
        """Calculate an estimate of the F-measure based on the scores"""
        num = np.sum(theta * weights * predictions)
        den = np.sum(theta * weights * (1 - alpha) + \
                     alpha * predictions * weights)
        F_guess = 0.5 if den==0 else num/den
        return F_guess

    def _calc_inst_pmf(self):
        """Calculate the epsilon-greedy instrumental distribution"""
        # Easy vars
        t = self.t_
        epsilon = self.epsilon
        alpha = self.alpha
        predictions = self.strata.mean_pred
        weights = self.strata.weights_
        p1 = self._BB_model.theta_
        p0 = 1 - p1
        F = np.nan if t == 0 else self.estimate_[t - 1]

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
            self.inst_pmf_[:,t] = inst_pmf
        else:
            self.inst_pmf_ = inst_pmf

    def reset(self):
        """Resets the sampler to its initial state

        Note
        ----
        This will destroy the label cache, instrumental distribution and
        history of estimates.
        """
        super(OASISSampler, self).reset()
        self.strata.reset()
        self._BB_model.reset()

        # Array to record history of instrumental distributions
        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.num_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.num_strata_, dtype=float)
