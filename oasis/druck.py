import numpy as np
from scipy.special import expit
import warnings
import copy

from .passive import PassiveSampler
from .stratification import auto_stratify
from .oasis import BetaBernoulliModel
from .input_verification import (verify_scores, verify_consistency, \
                                 verify_strata, scores_to_probs)

class DruckSampler(PassiveSampler):
    """Stratified sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an adaptive instrumental
    distribution that minimises asymptotic variance. See reference
    [Druck2011]_ for details.

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

    scores : array-like, shape=(n_items,)
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

    strata : Strata instance, optional, default None
        Describes how to stratify the pool. If not given, the stratification
        will be done automatically based on the scores given. Additional
        keyword arguments may be passed to control this automatic
        stratification (see below).

    max_iter : int, optional, default None
        Maximum number of iterations to expect for pre-allocating arrays.
        Once this limit is reached, sampling can no longer continue. If no
        value is given, defaults to n_items.

    replace : boolean, optional, default True
        Whether to sample with or without replacement.

    Other Parameters
    ----------------
    opt_class : array-like, dtype=bool, shape=(n_class,), optional, default None
        Indicates which classifier scores to use when stratifying the pool (if
        `strata` is None). If opt_class is False for a classifier, then its
        scores will not be used in calculating the strata, however estimates of
        its performance will still be calculated.

    identifiers : array-like, optional, default None
        Unique identifiers for the items in the pool. Must match the row order
        of the "predictions" parameter. If no value is given, defaults to
        [0, 1, ..., n_items].

    debug : bool, optional, default False
        Whether to print out verbose debugging information.

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

    References
    ----------
    .. [Druck2011] G. Druck and A. McCallum, "Toward Interactive Training and
       Evaluation," in Proceedings of the 20th ACM International Conference on
       Information and Knowledge Management, 2011, pp. 947–956.

    """
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 opt_class=None, strata=None, max_iter=None, identifiers=None,
                 replace=True, debug=False, **kwargs):
        super(DruckSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, identifiers, replace, debug)
        self.scores = verify_scores(scores)
        self.proba, self.opt_class = \
            verify_consistency(self.predictions, self.scores, proba, opt_class)
        self.strata = verify_strata(strata)

        #: Generate strata if not given
        if self.strata is None:
            # Need to transform scores to a common range [0,1] (so that we can
            # average them)
            self._probs = scores_to_probs(self.scores, self.proba)

            if np.sum(self.opt_class) > 1:
                # Average the probabilities over opt_class
                self._probs_avg_opt_class = np.mean(self._probs[:,self.opt_class], \
                                                     axis=1, keepdims=True)
                # If optimising over multiple classifiers, use the averaged
                # probabilities to stratify
                self.strata = \
                    auto_stratify(self._probs_avg_opt_class.ravel(), **kwargs)
            else:
                # Otherwise use scores from single classifier to stratify
                self.strata = \
                    auto_stratify(self.scores[:,self.opt_class].ravel(), \
                                  **kwargs)
        else:
            self.strata.reset()
        
        #: Number of TP, PP, P sampled per stratum
        self._TP_st = np.zeros([self.strata.n_strata_, self._n_class])
        self._PP_st = np.zeros([self.strata.n_strata_, self._n_class])
        self._P_st = np.zeros([self.strata.n_strata_, self._n_class])

        #: Rate at which TP, PP, P are sampled per stratum
        self._TP_rates = np.zeros([self.strata.n_strata_, self._n_class])
        self._PP_rates = np.zeros([self.strata.n_strata_, self._n_class])
        self._P_rates = np.zeros([self.strata.n_strata_, self._n_class])

        #: Initialise with two samples from each stratum
        #TODO: Fails if stratum contains only one item. Fix at reset also.
        for k in self.strata.indices_:
            for i in range(2):
                self._iterate(fixed_stratum = k, calc_rates = False)

    def _sample_item(self, **kwargs):
        """Sample an item from the strata"""
        if 'fixed_stratum' in kwargs:
            stratum_idx = kwargs['fixed_stratum']
        else:
            stratum_idx = None
        if stratum_idx is not None:
            #: Sample in given stratum
            loc = self.strata._sample_in_stratum(stratum_idx,
                                                 replace=self.replace)
        else:
            loc, stratum_idx = self.strata.sample(pmf = self.strata.weights_,
                                                  replace=self.replace)

        return loc, 1, {'stratum': stratum_idx}

    def _calc_estimate(self, TP_rates, PP_rates, P_rates, return_num_den=False):
        """
        """
        #: Easy vars
        sizes = self.strata.sizes_
        alpha = self.alpha
        #: Estimate number of TP, PP, P
        TP = np.inner(TP_rates.T, sizes)
        PP = np.inner(PP_rates.T, sizes)
        P = np.inner(P_rates.T, sizes)
        return self._F_measure(alpha, TP, PP - TP, P - TP, \
                               return_num_den=return_num_den)

    def _update_estimate_and_sampler(self, ell, ell_hat, weight, extra_info,
                                     **kwargs):
        """Update the estimate after querying the label for an item"""
        # Overwriting the base class method, since estimator is stratified
        
        stratum_idx = extra_info['stratum']

        if 'calc_rates' in kwargs:
            calc_rates = kwargs['calc_rates']
        else:
            calc_rates = True

        self._TP_st[stratum_idx,:] += ell_hat * ell * weight
        self._PP_st[stratum_idx,:] += ell_hat * weight
        self._P_st[stratum_idx,:] += ell * weight

        if calc_rates:
            self._P_rates = self._P_st/self.strata._n_sampled[:,np.newaxis]
            self._TP_rates = self._TP_st/self.strata._n_sampled[:,np.newaxis]
            self._PP_rates = self._PP_st/self.strata._n_sampled[:,np.newaxis]

        #: Update model estimate (with prior)
        self._estimate[self.t_], self._F_num, self._F_den = \
            self._calc_estimate(self._TP_rates, self._PP_rates, \
                                self._P_rates, return_num_den=True)

    def reset(self):
        """Resets the sampler to its initial state

        Note
        ----
        This will destroy the label cache and history of estimates
        """
        super(DruckSampler, self).reset()
        self.strata.reset()
        self._TP_st[:,:] = 0
        self._PP_st[:,:] = 0
        self._P_st[:,:] = 0
        self._TP_rates[:,:] = 0
        self._PP_rates[:,:] = 0
        self._P_rates[:,:] = 0

        for k in self.strata.indices_:
            for i in range(2):
                self._iterate(fixed_stratum = k, calc_rates = False)
