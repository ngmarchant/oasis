from typing import Callable, Tuple
import numpy as np
import warnings


class PassiveSampler:
    def __init__(self, alpha: float,
                 predictions, # 1d numpy array
                 scores, # 1d numpy array
                 oracle: Callable,
                 debug=False,
                 max_iter=None):

        self.alpha = alpha
        self.oracle = oracle
        self.predictions = predictions
        self.n_items = len(self.predictions)
        self.max_iter = self.n_items if (max_iter is None) else int(max_iter)
        self.identifiers = np.arange(self.n_items)
        self.debug = debug
        self.TP, self.FP, self.FN, self.idx, self.queried_oracle, self.cached_labels = [None] * 6
        self.reset()

    def estimate(self):
        return self.stored_estimates[0:self.idx]

    def reset(self):
        # Type I/II error terms
        self.TP = 0
        self.FP = 0
        self.FN = 0

        # Iteration index
        self.idx = 0

        # Array to record whether oracle was queried at each iteration
        self.queried_oracle = np.repeat(False, self.max_iter)
        self.cached_labels = np.repeat(np.nan, self.n_items)

        # Array to record history of F-measure estimates
        self.stored_estimates = np.repeat(np.nan, self.max_iter)

    def sample(self, n_to_sample: int, sample_with_replacement=True, **kwargs):
        """Sample a sequence of items from the pool (with replacement)

        Parameters
        ----------
        n_to_sample : positive int
            number of items to sample
        """
        for _ in range(n_to_sample):
            loc, weight = self._sample_item(sample_with_replacement, **kwargs)
            # Query label
            ell = self._query_label(loc)
            # Get predictions
            ell_hat = self.predictions[loc]

            if self.debug:
                print("Sampled label {} for item {}.".format(ell, loc))

            # Update
            self._update_estimate_and_sampler(ell, ell_hat, weight)

            self.idx += 1

    def sample_distinct(self, n_to_sample, **kwargs):
        self.sample(n_to_sample, sample_with_replacement=False)
   
    def _sample_item(self, sample_with_replacement: bool, **kwargs) -> Tuple:
        """Sample an item from the pool"""
        if sample_with_replacement:
            # Can sample from any of the items
            loc = np.random.choice(self.n_items)
        else:
            # Can only sample from items that have not been seen
            # Find ids that haven't been seen yet
            not_seen_ids, = np.where(np.isnan(self.cached_labels))
            #print("n:", not_seen_ids)
            if len(not_seen_ids) == 0:
                raise ValueError("all have been sampled")
            loc = np.random.choice(not_seen_ids)
        return loc, 1

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

        ell = self.cached_labels[loc]
        if not np.isnan(ell):
            return ell

            # Label has not been cached. Need to query oracle
        ell = self.oracle(loc)
        if ell not in [0, 1]:
            raise Exception("Oracle provided an invalid label.")
        #TODO Gracefully handle errors from oracle?
        self.queried_oracle[self.idx] = True
        self.cached_labels[loc] = ell

        return ell

    def _compute_f_score(self):
        """Calculate the weighted F-measure"""
        num = self.TP
        den = self.alpha * (self.TP + self.FP) + (1 - self.alpha) * (self.TP + self.FN)
        with np.errstate(divide='ignore', invalid='ignore'):
            return num/den

    def _update_estimate_and_sampler(self, ell, ell_hat, weight):
        """Update the estimate after querying the label for an item"""
        self.TP += ell_hat * ell * weight
        self.FP += ell_hat * (1 - ell) * weight
        self.FN += (1 - ell_hat) * ell * weight
        self.stored_estimates[self.idx] = self._compute_f_score()