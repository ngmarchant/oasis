import numpy as np
from scipy.special import expit
import copy
import warnings

from .passive import (PassiveSampler, verify_scores, verify_consistency)
from .oasis import BetaBernoulliModel
from .stratification import (Strata, stratify_by_features, stratify_by_scores,
                             auto_stratify)

class KadaneSampler(PassiveSampler):
    """

    """
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 epsilon=1e-3, prior_strength=None, decaying_prior=True,
                 strata=None, record_inst_hist=False, max_iter=None,
                 identifiers=None, debug=False, **kwargs):
        n_items = len(predictions)
        if max_iter is None:
            max_iter = n_items
        if (max_iter > n_items):
            warnings.warn("Setting max_iter to the size of the pool since "
                          "sampling without replacement.".format(n_items))
            max_iter = n_items
        super(KadaneSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, identifiers, debug)
        self.scores = scores
        self.proba = verify_proba(scores, proba)
        self.prior_strength = prior_strength
        self.epsilon = epsilon
        self.strata = strata
        self.record_inst_hist = record_inst_hist
        self.decaying_prior = decaying_prior

        # Delete variables from the base class which are not needed (because
        # this method uses a stratified estimator for the F-measure)
        del self._TP, self._FP, self._FN, self._TN

        # Deprecated arguments: n_init, method
        #self.n_init = n_init
        #self.method = method

        # Generate strata if not given
        if self.strata is None:
            self.strata = auto_stratify(self.scores, **kwargs)

        # Convert scores to probabilities if necessary
        if self.proba:
            probs = self.scores
        else:
            probs = expit(self.scores)

        # Quantities for generating priors
        self.strata.mean_prob = self.strata.intra_mean(probs)
        self.strata.mean_pred = self.strata.intra_mean(self.predictions)
        self.strata.mean_pred_prob = self.strata.intra_mean(self.predictions * probs)

        # Choose prior strength if not given
        if self.prior_strength is None:
            self.prior_strength = 2*self.strata.n_strata_

        # Instantiate Beta-Bernoulli models
        gamma_TP = self._calc_BB_prior(self.strata.mean_pred_prob, self.prior_strength)
        gamma_PP = self._calc_BB_prior(self.strata.mean_pred, self.prior_strength)
        gamma_P = self._calc_BB_prior(self.strata.mean_prob, self.prior_strength)
        self._BB_TP = BetaBernoulliModel(gamma_TP[0], gamma_TP[1], \
                                         decaying_prior = self.decaying_prior,
                                         store_wp=False)
        self._BB_PP = BetaBernoulliModel(gamma_PP[0], gamma_PP[1], \
                                         decaying_prior = self.decaying_prior,
                                         store_wp=False)
        self._BB_P = BetaBernoulliModel(gamma_P[0], gamma_P[1], \
                                         decaying_prior = self.decaying_prior,
                                         store_wp=False)

        # Array to record asymptotic variance estimate
        self._estimatevar_ = np.repeat(np.nan, self._max_iter)
        self._cov = np.zeros([3,3])
        self._grad_F = np.zeros(3)

        # Covariance between t, p, y variables for each stratum (3rd dimension)
        self.cov_model_ = np.zeros([self.strata.n_strata_,3,3])
        self._update_cov_model()

        # Array to store expected decrease in variance (for each stratum) if an
        #: additional item were sampled
        self._var_decrease = np.zeros(self.strata.n_strata_, dtype=float)

        # Update estimate (relying on prior for now)
        self._update_estimates()

        # Array to record history of instrumental distributions
        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.n_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.n_strata_, dtype=float)

        # Initialise with one sample from each stratum
        for k in self.strata.indices_:
            for i in range(1):
                self._iterate(fixed_stratum = k)

    def _sample_item(self, **kwargs):
        """Sample an item from the pool according to the instrumental
        distribution
        """
        t = self.t_
        if 'fixed_stratum' in kwargs:
            stratum_idx = kwargs['fixed_stratum']
        else:
            stratum_idx = None
        if stratum_idx is not None:
            # Sample in given stratum
            loc = self.strata._sample_in_stratum(stratum_idx, replace=False)
            # Record instrumental distribution
            if self.record_inst_hist:
                self.inst_pmf_[stratum_idx,t] = 1
        else:
            # Choose stratum based on instrumental distribution
            self._calc_inst_pmf()

            if self.record_inst_hist:
                inst_pmf = self.inst_pmf_[:,t]
            else:
                inst_pmf = self.inst_pmf_

            loc, stratum_idx = self.strata.sample(pmf = inst_pmf, replace=False)

        return loc, 1, {'stratum': stratum_idx}

    def _update_estimate_and_sampler(self, ell, ell_hat, weight, extra_info,
                                     **kwargs):
        """Update the BB models and the estimates"""
        stratum_idx = extra_info['stratum']
        self._BB_TP.update(ell*ell_hat, stratum_idx)
        self._BB_PP.update(ell_hat, stratum_idx)
        self._BB_P.update(ell, stratum_idx)

        # Update model covariance matrix for stratum_idx
        self._update_cov_model(strata_to_update = [stratum_idx])

        # Update F-measure estimate, estimator variance, exp. variance decrease
        self._update_estimates()

    def _calc_BB_prior(self, theta_0, prior_strength):
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

        #weighted_strength = self.weights * strength
        n_strata = len(theta_0)
        weighted_strength = prior_strength / n_strata
        alpha_0 = theta_0 * weighted_strength
        beta_0 = (1 - theta_0) * weighted_strength
        return alpha_0, beta_0

    def _update_estimates(self):
        """ """
        #: Easy vars
        t = self.t_
        weights = self.strata.weights_
        sizes = self.strata.sizes_
        alpha = self.alpha
        n_sampled = np.clip(self.strata._n_sampled, 2, np.inf)

        TP = np.dot(weights, self._BB_TP.theta_)
        PP = np.dot(weights, self._BB_PP.theta_)
        P = np.dot(weights, self._BB_P.theta_)
        cov_model = self.cov_model_

        #: Estimate of F
        self._estimate[t], self._F_num, self._F_den = \
            self._F_measure(alpha, TP, PP - TP, P - TP, return_num_den=True)

        #: Estimate of grad F w.r.t. Omega = (TP, PP, P)
        self._grad_F[0] = 1/self._F_den
        self._grad_F[1] = - alpha * self._F_num * self._grad_F[0]**2
        self._grad_F[2] = - (1 - alpha) * self._F_num * self._grad_F[0]**2

        #: Estimate of cov(omega)
        factor = weights**2 * (sizes - n_sampled)/(sizes * n_sampled)
        self._cov = np.tensordot(factor, cov_model, axes=1)

        #: Estimate of variance of F estimator
        self._estimatevar_[t] = \
            np.dot(self._grad_F, np.dot(self._cov, self._grad_F))

        #: Expected decrease in variance (for each stratum) if an additional
        #: item were sampled
        #self._var_decrease = ( weights**2 * (1/(n_sampled*(n_sampled+1))) *
        #                               ( cov_model[:,0,0] / self._F_den**2
        #                                 - 2 * self._F_num * (alpha * cov_model[:,0,1] + (1 - alpha) * cov_model[:,0,2]) / self._F_den**3
        #                                 + (self._F_num/self._F_den**2)**2 * ( (1 - alpha)**2 * cov_model[:,2,2] + alpha**2 * cov_model[:,1,1] + 2 * (1 - alpha) * alpha * cov_model[:,1,2]) ) )
        self._var_decrease = weights**2 * (1/(n_sampled*(n_sampled+1))) * \
            np.dot(np.tensordot(cov_model, self._grad_F, axes=1), self._grad_F)
        #: Special case: when stratum is fully sampled, cannot decrease
        #: variance further
        self._var_decrease[n_sampled == sizes] = 0

    def _update_cov_model(self, strata_to_update='all'):
        """
        strata_to_update : array-like or 'all'
            array containing stratum indices to update
        """
        if strata_to_update == 'all':
            strata_to_update = self.strata.indices_
        #: Otherwise assume strata_to_update is valid (no duplicates etc.)

        #: Update covariance matrices
        #: We usually update only one stratum at a time, so for loop is ok
        n_sampled = np.clip(self.strata._n_sampled, 2, np.inf) #: adding 2 avoids undef. cov
        factor = n_sampled/(n_sampled - 1)
        for k in strata_to_update:
            TP = self._BB_TP.theta_[k]
            PP = self._BB_PP.theta_[k]
            P = self._BB_P.theta_[k]

            self.cov_model_[k,0,0] = factor[k] * TP * (1 - TP)
            self.cov_model_[k,0,1] = factor[k] * TP * (1 - PP)
            self.cov_model_[k,0,2] = factor[k] * TP * (1 - P)
            self.cov_model_[k,1,1] = factor[k] * PP * (1 - PP)
            self.cov_model_[k,1,2] = factor[k] * (TP - PP * P)
            self.cov_model_[k,2,2] = factor[k] * P * (1 - P)
            self.cov_model_[k,1,0] = self.cov_model_[k,0,1]
            self.cov_model_[k,2,0] = self.cov_model_[k,0,2]
            self.cov_model_[k,2,1] = self.cov_model_[k,1,2]

    def _calc_inst_pmf(self):
        """
        """
        t = self.t_
        alpha = self.alpha
        #weights = self.strata.weights_
        epsilon = self.epsilon
        var_decrease = self._var_decrease
        estimate_var = self._estimatevar_[0] if t == 0 else self._estimatevar_[t-1]

        #: Expected relative decrease in variance (per stratum)
        rel_var_decrease = var_decrease/estimate_var

        #: Calculate argument of exp, rescaling to avoid overflow
        arg = rel_var_decrease/epsilon
        arg = arg - np.max(arg)

        #: Softmax distribution based on expected relative decrease in variance
        inst_pmf = np.exp(arg)
        #: Don't sample from strata if no variance reduction is possible
        inst_pmf = inst_pmf/np.sum(inst_pmf)

        #: Epsilon greedy
        #inst_pmf = epsilon * self.strata.weights_ + \
        #           (1 - epsilon) * self._var_decrease/np.sum(self._var_decrease)

        if self.record_inst_hist:
            self.inst_pmf_[:,t] = inst_pmf
        else:
            self.inst_pmf_ = inst_pmf

    def decision_split(self, stratum_idx):
        def variance_term(grad_F, cov_model, n_sampled, weight, size):
            #: Calculate variance contribution associated with a particular
            # stratum
            factor = weight**2 * (size - n_sampled)/(size * n_sampled)
            return factor * np.dot(grad_F, np.dot(cov_model, grad_F))

        #: Record the sampled locations in the stratum
        sampled_locs = np.where(self.strata._sampled[stratum_idx])[0]

        #: Number of possible splits
        num_splits = len(sampled_locs) - 3

        if num_splits <= 0:
            #: Can't split -- need at least two items in each child
            return None

        #: Array to store variance reduction for each possible split
        var_reductions = np.empty(num_splits)

        for i in range(2, 2 + num_splits):
            #: Calculate average t, p, y separately
            # Have to re-calculate everything, because the current data
            # structure only keeps track of the sufficient statistics (not
            # the individual t, p, y's)
            locs = [sampled_locs[0:i], sampled_locs[i::]]
            n_sampled = [len(x) for x in locs]
            size = [sampled_locs[i] + 1, self.strata.sizes_[stratum_idx] - (sampled_locs[i] + 1)]
            weight = [x/self.strata.n_items_ for x in size]
            labels = [self.cached_labels_[self.strata.allocations_[stratum_idx][x]] for x in locs]
            preds = [self.predictions[self.strata.allocations_[stratum_idx][x]] for x in locs]
            t_avg = [np.mean(preds[j] * labels[j]) for j in range(2)]
            p_avg = [np.mean(preds[j]) for j in range(2)]
            y_avg = [np.mean(labels[j]) for j in range(2)]

            #print("Split location {}".format(sampled_locs[i]))
            #print("n_sampled: {}".format(n_sampled))
            #print("size: {}".format(size))
            #print("weight: {}".format(weight))
            #print(labels)
            #print(preds)
            #print(t_avg)
            #print(p_avg)
            #print(y_avg)

            factor = [x/(x-1) for x in n_sampled]

            cov_model = [np.empty([3,3]), np.empty([3,3])]

            for j in range(2):
                cov_model[j][0,0] = factor[j] * t_avg[j] * (1 - t_avg[j])
                cov_model[j][0,1] = factor[j] * t_avg[j] * (1 - p_avg[j])
                cov_model[j][0,2] = factor[j] * t_avg[j] * (1 - y_avg[j])
                cov_model[j][1,1] = factor[j] * p_avg[j] * (1 - p_avg[j])
                cov_model[j][1,2] = factor[j] * (t_avg[j] - p_avg[j] * y_avg[j])
                cov_model[j][2,2] = factor[j] * y_avg[j] * (1 - y_avg[j])
                cov_model[j][1,0] = cov_model[j][0,1]
                cov_model[j][2,0] = cov_model[j][0,2]
                cov_model[j][2,1] = cov_model[j][1,2]


            TP_rates = copy.deepcopy(self._BB_TP.theta_)
            TP_rates[stratum_idx] = t_avg[0]
            TP_rates = np.append(TP_rates, t_avg[1])
            PP_rates = copy.deepcopy(self._BB_PP.theta_)
            PP_rates[stratum_idx] = p_avg[0]
            PP_rates = np.append(PP_rates, p_avg[1])
            P_rates = copy.deepcopy(self._BB_P.theta_)
            P_rates[stratum_idx] = y_avg[0]
            P_rates = np.append(P_rates, y_avg[1])
            weights = copy.deepcopy(self.strata.weights_)
            weights[stratum_idx] = weight[0]
            weights = np.append(weights, weight[1])
            TP = np.dot(weights, TP_rates)
            PP = np.dot(weights, PP_rates)
            P = np.dot(weights, P_rates)

            alpha = self.alpha
            estimate_new, F_num_new, F_den_new = \
            self._F_measure(alpha, TP, PP - TP, P - TP, return_num_den=True)

            #: Estimate of grad F w.r.t. Omega = (TP, PP, P)
            grad_F_new = np.empty(3)
            grad_F_new[0] = 1/F_den_new
            grad_F_new[1] = - alpha * F_num_new * grad_F_new[0]**2
            grad_F_new[2] = - (1 - alpha) * F_num_new * grad_F_new[0]**2

            #: Compute variance reduction. Will be variance term from stratum_idx
            # minus variance terms calculated for the separate strata above.
            #: TODO grad_F actually differs pre and post split
            var_contrib_old = variance_term(self._grad_F, \
                                          self.cov_model_[stratum_idx,:,:], \
                                          self.strata._n_sampled[stratum_idx], \
                                          self.strata.weights_[stratum_idx], \
                                          self.strata.sizes_[stratum_idx])
            var_contrib_l = variance_term(grad_F_new, \
                                          cov_model[0], \
                                          n_sampled[0], \
                                          weight[0], \
                                          size[0])
            var_contrib_r = variance_term(grad_F_new, \
                                          cov_model[1], \
                                          n_sampled[1], \
                                          weight[1], \
                                          size[1])
            var_reductions[i-2] = var_contrib_old - (var_contrib_l + var_contrib_r)
        return var_reductions

    def reset(self):
        super(KadaneSampler, self).reset()
        self.strata.reset()
        self._BB_TP.reset()
        self._BB_PP.reset()
        self._BB_P.reset()

        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.n_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.n_strata_, dtype=float)

        self._estimatevar_ = np.repeat(np.nan, self._max_iter)
        self._cov = np.zeros([3,3])
        self._grad_F = np.zeros(3)

        self.cov_model_ = np.zeros([self.strata.n_strata_,3,3])
        self._update_cov_model()

        self._var_decrease = np.zeros(self.strata.n_strata_, dtype=float)

        self._update_estimates()

        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.n_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.n_strata_, dtype=float)

        for k in self.strata.indices_:
            for i in range(1):
                self._iterate(fixed_stratum = k)

class OptKadaneSampler(KadaneSampler):
    """
    Assumes that the covariance matrices and F-measure are known in advance
    """
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 prior_strength=None, decaying_prior=True,
                 strata=None, record_inst_hist=False, max_iter=None,
                 identifiers=None, debug=False, **kwargs):
        n_items = len(predictions)
        if max_iter is None:
            max_iter = n_items
        if (max_iter > n_items):
            warnings.warn("Setting max_iter to the size of the pool since "
                          "sampling without replacement.".format(n_items))
            max_iter = n_items
        super(KadaneSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, identifiers, debug)
        self.scores = scores
        self.proba = verify_proba(scores, proba)
        self.prior_strength = prior_strength
        self.strata = strata
        self.record_inst_hist = record_inst_hist
        self.decaying_prior = decaying_prior
        self._requires_updating = True

        # Delete variables from the base class which are not needed (because
        # this method uses a stratified estimator for the F-measure)
        del self._TP, self._FP, self._FN, self._TN

        # Deprecated arguments: n_init, method
        #self.n_init = n_init
        #self.method = method

        # Generate strata if not given
        if self.strata is None:
            self.strata = auto_stratify(self.scores, **kwargs)

        # Generate probabilities from scores if necessary
        if self.proba:
            probs = self.scores
        else:
            probs = expit(self.scores)

        # Quantities for generating priors
        self.strata.mean_prob = self.strata.intra_mean(probs)
        self.strata.mean_pred = self.strata.intra_mean(self.predictions)
        self.strata.mean_pred_prob = self.strata.intra_mean(self.predictions * probs)

        # Choose prior strength if not given
        if self.prior_strength is None:
            self.prior_strength = 2*self.strata.n_strata_

        # Instantiate Beta-Bernoulli models
        gamma_TP = self._calc_BB_prior(self.strata.mean_pred_prob, self.prior_strength)
        gamma_PP = self._calc_BB_prior(self.strata.mean_pred, self.prior_strength)
        gamma_P = self._calc_BB_prior(self.strata.mean_prob, self.prior_strength)
        self._BB_TP = BetaBernoulliModel(gamma_TP[0], gamma_TP[1], \
                                         decaying_prior = self.decaying_prior,
                                         store_wp=False)
        self._BB_PP = BetaBernoulliModel(gamma_PP[0], gamma_PP[1], \
                                         decaying_prior = self.decaying_prior,
                                         store_wp=False)
        self._BB_P = BetaBernoulliModel(gamma_P[0], gamma_P[1], \
                                         decaying_prior = self.decaying_prior,
                                         store_wp=False)

        # Array to record asymptotic variance estimate
        self.true_var_ = np.repeat(np.nan, self._max_iter)
        self._cov = np.zeros([3,3])
        self._true_grad_F = np.zeros(3)

        # Covariance between t, p, y variables for each stratum (3rd dimension)
        self.cov_model_ = np.zeros([self.strata.n_strata_,3,3])


        # Array to store expected decrease in variance (for each stratum) if an
        #: additional item were sampled
        self._var_decrease = np.zeros(self.strata.n_strata_, dtype=float)

        # Cache all labels
        for i,j in enumerate(self.identifiers):
            self.cached_labels_[i] = oracle(j)
        self._calc_true_parameters()

        # Update estimate (relying on prior for now)
        self._update_estimates()

        # Array to record history of instrumental distributions
        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.n_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.n_strata_, dtype=float)

        # Initialise with one sample from each stratum
        for k in self.strata.indices_:
            for i in range(1):
                self._iterate(fixed_stratum = k)

    def _calc_true_parameters(self):
        self._true_TP = self.strata.intra_mean(self.predictions * self.cached_labels_)
        self._true_PP = self.strata.intra_mean(self.predictions)
        self._true_P = self.strata.intra_mean(self.cached_labels_)

        alpha = self.alpha
        sizes = self.strata.sizes_
        weights = self.strata.weights_

        factor = sizes/(sizes - 1)
        for k in self.strata.indices_:
            TP = self._true_TP[k]
            PP = self._true_PP[k]
            P = self._true_P[k]

            self.cov_model_[k,0,0] = factor[k] * TP * (1 - TP)
            self.cov_model_[k,0,1] = factor[k] * TP * (1 - PP)
            self.cov_model_[k,0,2] = factor[k] * TP * (1 - P)
            self.cov_model_[k,1,1] = factor[k] * PP * (1 - PP)
            self.cov_model_[k,1,2] = factor[k] * (TP - PP * P)
            self.cov_model_[k,2,2] = factor[k] * P * (1 - P)
            self.cov_model_[k,1,0] = self.cov_model_[k,0,1]
            self.cov_model_[k,2,0] = self.cov_model_[k,0,2]
            self.cov_model_[k,2,1] = self.cov_model_[k,1,2]

        TP = np.dot(weights, self._true_TP)
        PP = np.dot(weights, self._true_PP)
        P = np.dot(weights, self._true_P)

        self._true_estimate, self._true_F_num, self._true_F_den = \
            self._F_measure(alpha, TP, PP - TP, P - TP, return_num_den=True)

        #: Estimate of grad F w.r.t. Omega = (TP, PP, P)
        self._true_grad_F[0] = 1/self._true_F_den
        self._true_grad_F[1] = - alpha * self._true_F_num * self._true_grad_F[0]**2
        self._true_grad_F[2] = - (1 - alpha) * self._true_F_num * self._true_grad_F[0]**2

    def _update_cov_model(self, strata_to_update='all'):
        return

    def _update_estimates(self):
        """ """
        #: Easy vars
        t = self.t_
        weights = self.strata.weights_
        sizes = self.strata.sizes_
        alpha = self.alpha
        n_sampled = np.clip(self.strata._n_sampled, 2, np.inf)

        TP = np.dot(weights, self._BB_TP.theta_)
        PP = np.dot(weights, self._BB_PP.theta_)
        P = np.dot(weights, self._BB_P.theta_)
        cov_model = self.cov_model_

        #: Estimate of F
        self._estimate[t], self._F_num, self._F_den = \
            self._F_measure(alpha, TP, PP - TP, P - TP, return_num_den=True)

        #: Estimate of cov(omega)
        factor = weights**2 * (sizes - n_sampled)/(sizes * n_sampled)
        self._cov = np.tensordot(factor, cov_model, axes=1)

        #: Estimate of variance of F estimator
        self.true_var_[t] = \
            np.dot(self._true_grad_F, np.dot(self._cov, self._true_grad_F))

        #: Expected decrease in variance (for each stratum) if an additional
        #: item were sampled
        self._var_decrease = weights**2 * (1/(n_sampled*(n_sampled+1))) * \
            np.dot(np.tensordot(cov_model, self._true_grad_F, axes=1), self._true_grad_F)
        #: Special case: when stratum is fully sampled, cannot decrease
        #: variance further
        self._var_decrease[n_sampled == sizes] = 0

    def _calc_inst_pmf(self):
        """
        """
        t = self.t_
        alpha = self.alpha
        #weights = self.strata.weights_
        var_decrease = self._var_decrease

        #: Softmax distribution based on expected relative decrease in variance
        inst_pmf = np.zeros(self.strata.n_strata_)
        best_stratum_idx = np.where(np.logical_and(var_decrease == np.max(var_decrease), self.strata.sizes_ - self.strata._n_sampled > 0))[0]
        if len(best_stratum_idx) > 1:
            best_stratum_idx = np.random.choice(best_stratum_idx)
        inst_pmf[best_stratum_idx] = 1

        #: Epsilon greedy
        #inst_pmf = epsilon * self.strata.weights_ + \
        #           (1 - epsilon) * self._var_decrease/np.sum(self._var_decrease)

        if self.record_inst_hist:
            self.inst_pmf_[:,t] = inst_pmf
        else:
            self.inst_pmf_ = inst_pmf

    def reset(self):
        super(KadaneSampler, self).reset()
        self.strata.reset()
        self._BB_TP.reset()
        self._BB_PP.reset()
        self._BB_P.reset()

        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.n_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.n_strata_, dtype=float)

        self.true_var_ = np.repeat(np.nan, self._max_iter)
        self._cov = np.zeros([3,3])

        self._var_decrease = np.zeros(self.strata.n_strata_, dtype=float)

        self._update_estimates()

        if self.record_inst_hist:
            self.inst_pmf_ = np.zeros([self.strata.n_strata_, self._max_iter],
                                     dtype=float)
        else:
            self.inst_pmf_ = np.zeros(self.strata.n_strata_, dtype=float)

        for k in self.strata.indices_:
            for i in range(1):
                self._iterate(fixed_stratum = k)
