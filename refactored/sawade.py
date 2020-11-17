import numpy as np

from refactored.passive import PassiveSampler


class ImportanceSampler(PassiveSampler):
    """Importance sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an instrumental
    distribution that minimises asymptotic variance. The instrumental
    distribution is estimated based on classifier confidence scores. True
    labels are queried from an oracle. See reference [Sawade2010]_ for details.

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

    epsilon : float, optional, default 1e-3
        Epsilon-greedy parameter. Valid values are on the interval [0, 1]. The
        "asymptotically optimal" distribution is sampled from with probability
        `1 - epsilon` and the passive distribution is sampled from with
        probability `epsilon`. The sampling is close to "optimal" for small
        epsilon.

    max_iter : int, optional, default None
        Maximum number of iterations to expect for pre-allocating arrays.
        Once this limit is reached, sampling can no longer continue. If no
        value is given, defaults to n_items.

    Other Parameters
    ----------------
    opt_class : array-like, dtype=bool, shape=(n_class,), optional, default None
        Indicates which classifiers to use in calculating the optimal
        distribution. If opt_class is False for a classifier, then its
        predictions and scores will not be used in calculating the optimal
        distribution, however estimates of its performance will still be
        calculated.

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

    inst_pmf_ : numpy.ndarray, shape=(n_items,)
        Epsilon-greedy instrumental pmf used for sampling.

    References
    ----------
    .. [Sawade2010] C. Sawade, N. Landwehr, and T. Scheffer, “Active Estimation
       of F-Measures,” in Advances in Neural Information Processing Systems 23,
       2010, pp. 2083–2091
    """
    def __init__(self, alpha, predictions, scores, oracle,
                 epsilon: float = 1e-3, max_iter=None):
        super(ImportanceSampler, self).__init__(alpha, predictions, scores, oracle,
                                          max_iter=max_iter)
        self.scores = scores
        self.epsilon = epsilon

        # Need to transform scores to the [0,1] interval (to use as proxy for
        # probabilities)
        # Average the probabilities over opt_class
        self._F_guess = self._calc_F_guess(self.alpha,
                                           self.predictions,
                                           self.scores)

        self._inst_pmf = np.empty(self.n_items, dtype=float)
        self._initialise_pmf()

    def _sample_item(self, sample_with_replacement:bool):
        """Sample an item from the pool according to the instrumental
        distribution
        """
        loc = np.random.choice(self.n_items, p=self._inst_pmf)
        weight = (1/self.n_items)/self._inst_pmf[loc]
        return loc, weight

    def _calc_F_guess(self, alpha, predictions, probabilities):
        """Calculate an estimate of the F-measure based on the scores"""
        num = np.sum(predictions * probabilities)
        den = np.sum((1 - alpha) * probabilities + alpha * predictions)
        F_guess = num / den
        # Ensure guess is not undefined
        if np.isnan(F_guess):
            F_guess = 0.5
        return F_guess

    def _initialise_pmf(self):
        """Calculate the epsilon-greedy instrumental distribution"""
        # Easy vars
        epsilon = self.epsilon
        alpha = self.alpha
        preds = self.predictions
        p1 = self.scores
        p0 = 1 - p1
        n_items = self.n_items
        F = self._F_guess

        # Calculate optimal instrumental pmf
        sqrt_arg = preds * (alpha**2 * F**2 * p0 + (1 - F)**2 * p1) + \
                          (1 - preds) * (1 - alpha)**2 * F**2 * p1

        self._inst_pmf = np.sqrt(sqrt_arg)
        # Normalize
        self._inst_pmf /= np.sum(self._inst_pmf)
        # Epsilon-greedy: (1 - epsilon) q + epsilon * p
        self._inst_pmf *= (1 - epsilon)
        self._inst_pmf += epsilon * 1 / n_items