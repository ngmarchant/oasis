import numpy as np
import warnings

def verify_proba(scores, proba):
    """
    Corrects the value of `proba` (if the scores don't match).
    """
    if (np.any(np.logical_or(scores < 0, scores > 1)) and proba):
        warnings.warn("Scores fall outside the [0,1] interval. Setting " +
                      "proba=False.")
        return False
    return proba

def is_pos_integer(number):
    """Checks whether argument is a positive integer"""
    if (type(number) is int) and (number > 0):
        return True
    else:
        return False

class BaseSampler:
    """Base class for all samplers"""
    def __init__(self, alpha, predictions, oracle, max_iter=None, indices=None,
                 debug=False):
        self.alpha = alpha
        self.oracle = oracle
        self.indices = indices
        self.predictions = predictions
        self._pool_size = len(predictions)
        self._max_iter = self._pool_size if (max_iter is None) else max_iter
        self.debug=debug
        self._requires_updating = False

        # Make item ids if not given
        if self.indices is None:
            self.indices = np.arange(self._pool_size)

        # Terms that make up the F-measure
        self._TP = 0
        self._PP = 0
        self._P = 0

        # Iteration index
        self.t_ = 0

        # Array to record whether oracle was queried at each iteration
        self.queried_oracle_ = np.repeat(False, self._max_iter)

        self.cached_labels_ = np.repeat(np.nan, self._pool_size)

        # Array to record history of F-measure estimates
        self.estimate_ = np.repeat(np.nan, self._max_iter)

    def reset(self):
        """Resets the sampler to its initial state

        Note
        ----
        This will destroy the label cache and history of estimates.
        """
        self._TP = 0
        self._PP = 0
        self._P = 0
        self.t_ = 0
        self.queried_oracle_ = np.repeat(False, self._max_iter)
        self.cached_labels_ = np.repeat(np.nan, self._pool_size)
        self.estimate_ = np.repeat(np.nan, self._max_iter)

    def _iterate(self):
        """Procedure for a single iteration (sampling and updating)"""
        # Sample item
        loc, weight, extra_info = self._sample_item()
        # Query label
        ell = self._query_label(loc)
        # Get prediction
        ell_hat = self.predictions[loc]

        if self.debug == True:
            print("Sampled label {} for item {}.".format(ell,loc))

        # Update estimate
        self._update_estimate(ell, ell_hat, weight)
        if self._requires_updating:
            self._update_sampler(ell, ell_hat, loc, weight, extra_info)

        self.t_ = self.t_ + 1

    def sample(self, n_items):
        """Sample a sequence of items from the pool

        Parameters
        ----------
        n_items : int
            number of items to sample
        """
        if not is_pos_integer(n_items):
            raise ValueError("n_items must be a positive integer.")

        n_remaining = self._max_iter - self.t_

        if n_remaining == 0:
            raise Exception("No more space available to continue sampling. "
                            "Consider re-initialising with a larger value "
                            "of max_iter.")

        if n_items > n_remaining:
            warnings.warn("Space only remains for {} more iteration(s). "
                          "Setting n_items = {}.".format(n_remaining, \
                          n_remaining))
            n_items = n_remaining

        for _ in range(n_items):
            self._iterate()

    def sample_distinct(self, n_items):
        """Sample a sequence of items from the pool until a minimum number of
        distinct items are queried

        Parameters
        ----------
        n_items : int
            number of distinct items to sample. If sampling with replacement,
            this number is not necessarily the same as the number of
            iterations.
        """
        # Record how many distinct items have not yet been sampled
        n_notsampled = np.sum(np.isnan(self.cached_labels_))

        if n_notsampled == 0:
            raise Exception("All distinct items have already been sampled.")

        if n_items > n_notsampled:
            warnings.warn("Only {} distinct item(s) have not yet been sampled."
                          " Setting n_items = {}.".format(n_notsampled, \
                          n_notsampled))
            n_items = n_notsampled

        n_sampled = 0 # number of distinct items sampled this round
        while n_sampled < n_items:
            self.sample(1)
            n_sampled += self.queried_oracle_[self.t_ - 1]*1

    def _sample_item(self):
        """Sample a single item from the pool. Varies depending on sampler."""
        return

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
            oracle_arg = self.indices[loc]
            ell = self.oracle(oracle_arg)
            if ell not in [0, 1]:
                raise Exception("Oracle provided an invalid label.")
            #TODO Gracefully handle errors from oracle?
            self.queried_oracle_[self.t_] = True
            self.cached_labels_[loc] = ell

        return ell

    def _F_measure(self, alpha, TP, PP, P):
        """Calculate the weighted F-measure"""
        num = TP
        den = (alpha * PP + (1 - alpha) * P)
        return np.nan if (den == 0) else (num/den)

    def _update_estimate(self, ell, ell_hat, weight):
        """Update the estimate after querying the label for an item"""
        if ell == 1 and ell_hat == 1:
            # Point is true positive
            self._TP = self._TP + weight
            self._PP = self._PP + weight
            self._P = self._P + weight
        elif ell_hat == 1:
            # Point is false positive
            self._PP = self._PP + weight
        elif ell == 1:
            # Point is false negative
            self._P = self._P + weight

        self.estimate_[self.t_] = \
                self._F_measure(self.alpha, self._TP, self._PP, self._P)

    def _update_sampler(self, ell, ell_hat, loc, weight, extra_info):
        """Used for adaptive samplers"""
        return
