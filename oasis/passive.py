import numpy as np
import warnings

from .base import (BaseSampler, is_pos_integer)

class PassiveSampler(BaseSampler):
    """
    Implements unbiased (passive) sampling to estimate alpha-weighted
    F-measures.

    Input
    -----
    labels : vector of ground truth labels (0 or 1). The position in the vector
              is assumed to be the identifier of the point.

    preds : vector of predictions (0/1 labels). The position in the vector is
            assumed to be the identifier of the point.

    alpha : double on the unit interval. Indicates the weight to use for
              the F-measure. alpha = 0.5 corresponds to the balanced F-measure,
              alpha = 1 corresponds to precision and alpha = 0 corresponds to
              recall. (default: 0.5)

    max_iter: positive integer. Maximum number of labels to be drawn. Used to
              preallocate arrays. (default: None)

    replace:  boolean. If True, sample with replacement, otherwise sample
              without replacement. (default: True)

    debug:    boolean. If True, print verbose information. (default: False)
    """
    def __init__(self, alpha, oracle, predictions, max_iter=None, indices=None,
                 replace=True, debug=False):
        self.replace = replace
        num_items = len(predictions)
        if not self.replace and max_iter > num_items:
            warnings.warn("Setting max_iter to the size of the pool since "
                          "sampling without replacement.".format(num_items))
            max_iter = num_items
        super(PassiveSampler, self).__init__(alpha, oracle, predictions,
                                             max_iter, indices, debug)

    def _sample_item(self):
        """
        Samples an item from the pool taking into account whether sampling with
        or without replacement.
        """
        if self.replace:
            # Can sample from any of the items
            loc = np.random.choice(self._num_items)
        else:
            # Can only sample from items that have not been seen
            # Find ids that haven't been seen yet
            not_seen_ids = np.where(np.isnan(self.cached_labels))[0]
            loc = np.random.choice(not_seen_ids)
        return loc, 1, {}

    def sample(self, n_items):
        """
        Sample `n_items`
        """
        if self.replace:
            super(PassiveSampler, self).sample(n_items)
        else:
            # Sampling without replacement changes the language used in the
            # exceptions.
            if not is_pos_integer(n_items):
                raise ValueError("n_items must be a positive integer.")

            n_remaining = self._max_iter - self.t

            if n_remaining == 0:
                if self._num_items == self._max_iter:
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
