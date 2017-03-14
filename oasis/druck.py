import numpy as np
import warnings

from .base import BaseSampler
from .passive import PassiveSampler
from .stratification import auto_stratify

class DruckSampler(PassiveSampler):
    """
    Input
    -----
    labels : int numpy array of length n
        array containing binary labels (assumed to be "0" or "1") for each
        data point.

    strata: an instance of the `Strata` class
        contains information about the data points (allocations, scores)
        and facilitates sampling from the strata.

    alpha : float, optional, default 0.5
        weight to use for the F-measure. It should be between 0 and 1, with a
        value of 1 corresponding to "precision", a value of 0 corresponding to
        "recall" and a value of 0.5 corresponding to the balanced F-measure
        (equal weight on precision and recall). Note that this parameterisation
        of the weighted F-measure is different to the usual one (see the paper
        cited above for the definition).

    replace : boolean
        If True, sample with replacement, otherwise, sample without
        replacement.

    debug : bool, optional, default False
        if True, prints debugging information.
    """
    def __init__(self, alpha, predictions, oracle, scores, strata=None,
                 max_iter=None, indices=None, replace=True, debug=False,
                 **kwargs):
        super(DruckSampler, self).__init__(alpha, predictions, oracle, 
                                           max_iter, indices, replace, debug)
        self.scores = scores
        self.strata = strata

        # Generate strata if not given
        if self.strata is None:
            self.strata = auto_stratify(self.scores, kwargs)

    def _sample_item(self):
        """
        Samples an item from the strata
        """
        # Sample label and record weight
        loc, stratum_idx = self.strata.sample(pmf = self.strata.weights)

        return loc, 1, {'stratum': stratum_idx}

    def reset(self):
        super(DruckSampler, self).reset()
        self.strata.reset()
