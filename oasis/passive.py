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
        super(PassiveSampler, self).__init__(alpha, predictions, oracle,
                                             max_iter, identifiers, replace, debug)

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
