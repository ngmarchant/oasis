import numpy as np
import warnings

from .input_verification import (verify_positive, verify_predictions, \
                                 verify_boolean, verify_identifiers, \
                                 verify_unit_interval)

class PassiveSampler:
    """Passive sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items uniformly and querying their labels from
    an oracle (which must be provided).

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

    oracle : function
        Function that returns ground truth labels for items in the pool. The
        function should take an item identifier as input (i.e. its
        corresponding row index) and return the ground truth label. Valid
        labels are 0 or 1.

    max_iter : int, optional, default None
        Maximum number of iterations to expect for pre-allocating arrays.
        Once this limit is reached, sampling can no longer continue. If no
        value is given, defaults to n_items.

    replace : bool, optional, default True
        Whether to sample with or without replacement.

    Other Parameters
    ----------------
    identifiers : array-like, optional, default None
        Unique identifiers for the items in the pool. Must match the row order
        of the "predictions" parameter. If no value is given, defaults to
        [0, 1, ..., n_items].

    debug : bool, optional, default True
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
    """
    def __init__(self, alpha, predictions, oracle, max_iter=None,
                 identifiers=None, replace=True, debug=False):
        self.alpha = verify_unit_interval(float(alpha))
        self.oracle = oracle
        self.predictions = verify_predictions(predictions)
        self._n_class = self.predictions.shape[1]
        self._multiple_class = True if self._n_class > 1 else False
        self._n_items = self.predictions.shape[0]
        self._max_iter = self._n_items if (max_iter is None) else int(max_iter)
        self.identifiers = verify_identifiers(identifiers, self._n_items)
        self.replace = verify_boolean(replace)
        self.debug = verify_boolean(debug)

        # If sampling without replacement, make sure max_iter is not
        # unnecessarily large
        if (not self.replace) and (self._max_iter > self._n_items):
            warnings.warn("Setting max_iter to the size of the pool since "
                          "sampling without replacement.".format(self._n_items))
            self._max_iter = self._n_items

        # Make item ids if not given
        if self.identifiers is None:
            self.identifiers = np.arange(self._n_items)

        # Type I/II error terms
        self._TP = np.zeros(self._n_class)
        self._FP = np.zeros(self._n_class)
        self._FN = np.zeros(self._n_class)
        self._TN = np.zeros(self._n_class)

        # Iteration index
        self.t_ = 0

        # Array to record whether oracle was queried at each iteration
        self._queried_oracle = np.repeat(False, self._max_iter)

        self.cached_labels_ = np.repeat(np.nan, self._n_items)

        # Array to record history of F-measure estimates
        self._estimate = np.tile(np.nan, [self._max_iter, self._n_class])

    @property
    def estimate_(self):
        if self.t_ == 0:
            return None
        if self._multiple_class:
            return self._estimate[0:self.t_,:]
        else:
            return self._estimate[0:self.t_,:].ravel()
    @estimate_.setter
    def estimate_(self, value):
        raise AttributeError("can't set attribute.")
    @estimate_.deleter
    def estimate_(self):
        raise AttributeError("can't delete attribute.")

    @property
    def queried_oracle_(self):
        if self.t_ == 0:
            return None
        else:
            return self._queried_oracle[0:self.t_]
    @queried_oracle_.setter
    def queried_oracle_(self, value):
        raise AttributeError("can't set attribute.")
    @queried_oracle_.deleter
    def queried_oracle_(self):
        raise AttributeError("can't delete attribute.")

    def reset(self):
        """Resets the sampler to its initial state

        Note
        ----
        This will destroy the label cache and history of estimates.
        """
        self._TP[:] = 0
        self._FP[:] = 0
        self._FN[:] = 0
        self._TN[:] = 0
        self.t_ = 0
        self._queried_oracle[:] = False
        self.cached_labels_[:] = np.nan
        self._estimate[:,:] = np.nan

    def _iterate(self, **kwargs):
        """Procedure for a single iteration (sampling and updating)"""
        # Sample item
        loc, weight, extra_info = self._sample_item(**kwargs)
        # Query label
        ell = self._query_label(loc)
        # Get predictions
        ell_hat = self.predictions[loc,:]

        if self.debug == True:
            print("Sampled label {} for item {}.".format(ell,loc))

        # Update
        self._update_estimate_and_sampler(ell, ell_hat, weight, extra_info, **kwargs)

        self.t_ = self.t_ + 1

    def sample(self, n_to_sample, **kwargs):
        """Sample a sequence of items from the pool

        Parameters
        ----------
        n_to_sample : int
            number of items to sample
        """
        n_to_sample = verify_positive(int(n_to_sample))

        n_remaining = self._max_iter - self.t_

        if n_remaining == 0:
            if (not self.replace) and (self._n_items == self._max_iter):
                raise Exception("All items have already been sampled")
            else:
                raise Exception("No more space available to continue sampling. "
                                "Consider re-initialising with a larger value "
                                "of max_iter.")

        if n_to_sample > n_remaining:
            warnings.warn("Space only remains for {} more iteration(s). "
                          "Setting n_to_sample = {}.".format(n_remaining, \
                          n_remaining))
            n_to_sample = n_remaining

        for _ in range(n_to_sample):
            self._iterate(**kwargs)

    def sample_distinct(self, n_to_sample, **kwargs):
        """Sample a sequence of items from the pool until a minimum number of
        distinct items are queried

        Parameters
        ----------
        n_to_sample : int
            number of distinct items to sample. If sampling with replacement,
            this number is not necessarily the same as the number of
            iterations.
        """
        # Record how many distinct items have not yet been sampled
        n_notsampled = np.sum(np.isnan(self.cached_labels_))

        if n_notsampled == 0:
            raise Exception("All distinct items have already been sampled.")

        if n_to_sample > n_notsampled:
            warnings.warn("Only {} distinct item(s) have not yet been sampled."
                          " Setting n_to_sample = {}.".format(n_notsampled, \
                          n_notsampled))
            n_to_sample = n_notsampled

        n_sampled = 0 # number of distinct items sampled this round
        while n_sampled < n_to_sample:
            self.sample(1,**kwargs)
            n_sampled += self._queried_oracle[self.t_ - 1]*1

    def _sample_item(self, **kwargs):
        """Sample an item from the pool"""
        if self.replace:
            # Can sample from any of the items
            loc = np.random.choice(self._n_items)
        else:
            # Can only sample from items that have not been seen
            # Find ids that haven't been seen yet
            not_seen_ids = np.where(np.isnan(self.cached_labels_))[0]
            loc = np.random.choice(not_seen_ids)
        return loc, 1, {}

    def _query_label(self, loc):
        """Query the label for the item with index `loc`. Preferentially
        queries the label from the cache, but if not yet cached, queries the
        oracle.

        Returns
        -------
        int
            the true label "0" or "1".
        """
        # Try to get label from cache
        ell = self.cached_labels_[loc]

        if np.isnan(ell):
            # Label has not been cached. Need to query oracle
            oracle_arg = self.identifiers[loc]
            ell = self.oracle(oracle_arg)
            if ell not in [0, 1]:
                raise Exception("Oracle provided an invalid label.")
            #TODO Gracefully handle errors from oracle?
            self._queried_oracle[self.t_] = True
            self.cached_labels_[loc] = ell

        return ell

    def _F_measure(self, alpha, TP, FP, FN, return_num_den=False):
        """Calculate the weighted F-measure"""
        num = np.float64(TP)
        den = np.float64(alpha * (TP + FP) + (1 - alpha) * (TP + FN))
        with np.errstate(divide='ignore', invalid='ignore'):
            F_measure = num/den
        #F_measure = num/den

        if return_num_den:
            return F_measure, num, den
        else:
            return F_measure

    def _update_estimate_and_sampler(self, ell, ell_hat, weight, extra_info,
                                     **kwargs):
        """Update the estimate after querying the label for an item"""
        self._TP += ell_hat * ell * weight
        self._FP += ell_hat * (1 - ell) * weight
        self._FN += (1 - ell_hat) * ell * weight
        self._TN += (1 - ell_hat) * (1 - ell) * weight

        self._estimate[self.t_] = \
                self._F_measure(self.alpha, self._TP, self._FP, self._FN)
