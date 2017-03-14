import numpy as np
import warnings

from .base import (BaseSampler, is_pos_integer)

class PassiveSampler(BaseSampler):
    """Passive sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items uniformly and querying their labels from
    an oracle (which must be provided).

    Parameters
    ----------
    alpha : float
        weight for the F-measure. Valid weights are on the interval [0, 1].
        ``alpha == 1`` corresponds to precision, ``alpha == 0`` corresponds to
        recall, and ``alpha == 0.5`` corresponds to the balanced F-measure.

    predictions : array-like, shape=(pool_size,)
        ordered array of predicted labels for each item in the pool. Valid
        labels are "0" or "1".

    oracle : function
        a function which takes an item id as input and returns the item's true
        label. Valid labels are "0" or "1".

    max_iter : int, optional, default None
        space for storing the sampling history is limited to a maximum
        number of iterations. Once this limit is reached, sampling can no
        longer continue. If no value is given, defaults to the size of
        the pool.

    indices : array-like, optional, default None
        ordered array of unique identifiers for the items in the pool.
        Should match the order of the "predictions" parameter. If no value is
        given, defaults to [0, 1, ..., pool_size].

    replace : bool, optional, default True
        whether to sample with or without replacement.

    debug : bool, optional, default False
        whether to print out verbose debugging information.

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
    """
    def __init__(self, alpha, predictions, oracle, max_iter=None, indices=None,
                 replace=True, debug=False):
        self.replace = replace
        pool_size = len(predictions)
        if max_iter is None:
            max_iter = pool_size
        if (not self.replace) and (max_iter > pool_size):
            warnings.warn("Setting max_iter to the size of the pool since "
                          "sampling without replacement.".format(pool_size))
            max_iter = pool_size
        super(PassiveSampler, self).__init__(alpha, predictions, oracle,
                                             max_iter, indices, debug)

    def _sample_item(self):
        """Sample an item from the pool"""
        if self.replace:
            # Can sample from any of the items
            loc = np.random.choice(self._pool_size)
        else:
            # Can only sample from items that have not been seen
            # Find ids that haven't been seen yet
            not_seen_ids = np.where(np.isnan(self.cached_labels_))[0]
            loc = np.random.choice(not_seen_ids)
        return loc, 1, {}

    def sample(self, n_items):
        """Sample a sequence of items from the pool

        Parameters
        ----------
        n_items : int
            number of items to sample
        """
        if self.replace:
            super(PassiveSampler, self).sample(n_items)
        else:
            # Sampling without replacement changes the language used in the
            # exceptions.
            if not is_pos_integer(n_items):
                raise ValueError("n_items must be a positive integer.")

            n_remaining = self._max_iter - self.t_

            if n_remaining == 0:
                if self._pool_size == self._max_iter:
                    raise Exception("All items have already been sampled.")
                else:
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
