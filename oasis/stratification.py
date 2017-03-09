import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import warnings
import copy

def stratify_by_features(features, num_strata, **kwargs):
    """
    Input
    -----
    features : float (num_items x num_fts) numpy.ndarray
        Feature matrix where rows correspond to data points and columns
        correspond to features.

    num_strata : integer
        Number of strata to create.

    preds : integer numpy.ndarray of length num_items, optional, default None
        Array of (0/1) label predictions for each datapoint. The location in
        the array is the datapoint index. If present, the datapoints
        are separated into two groups -- predicted positives and predicted
        negatives. Then points are allocated to strata separately within
        each group.

    **kwargs :
        Pass to sklearn.cluster.KMeans

    Output
    ------
    allocations : list of integer numpy.ndarrays
        This object is used to specify how the datapoints are allocated amongst
        a collection of strata. Each element of the list represents a stratum
        and points to an array of indices that specify the datapoints contained
        within the stratum.
    """

    n_items = features.shape[0]

    stratum_ids = np.empty(n_items, dtype='int')

    km = KMeans(n_clusters=num_strata, **kwargs)
    stratum_ids = km.fit_predict(X=features)

    # Convert to stratum allocations
    allocations = []
    for k in range(num_strata):
        allocations.append(np.where(stratum_ids == k)[0])

    return allocations

def _heuristic_bin_width(obs):
    """Calculates the optimal bin width to use for a histogram based on
    the Freedman-Diaconis rule.
    """
    IQR = sp.percentile(obs, 75) - sp.percentile(obs, 25)
    N = len(obs)
    return 2*IQR*N**(-1/3)

def stratify_by_scores(scores, goal_num_strata='auto', method='equal_size',
                       **kwargs):
    """
    Input
    -----
    scores : float numpy array of length n
        array containing scores for each data point.

    goal_num_strata : integer
        number of strata to create. For some methods, this number is
        a goal -- it is not guaranteed. The actual number of strata created is
        returned as output.

    method : 'cum_sqrt_F' or 'equal_size', optional, default 'equal_size'

    **kwargs : n_bins
               specify number of bins for cum sqrt F method. Otherwise use
               heuristic.
    Output
    ------
    allocations : list of integer numpy.ndarrays
        This object is used to specify how the datapoints are allocated amongst
        a collection of strata. Each element of the list represents a stratum
        and points to an array of indices that specify the datapoints contained
        within the stratum.
    """

    available_methods = ['equal_size', 'cum_sqrt_F']
    if method not in available_methods:
        raise ValueError("method argument is invalid")

    if (method == 'cum_sqrt_F') or (goal_num_strata == 'auto'):
        # computation below is needed for cum_sqrt_F method OR if we need to
        # determine the number of strata for equal_size method automatically
        if 'n_bins' in kwargs:
            n_bins = kwargs['n_bins']
        else:
            # choose n_bins heuristically
            width_score = _heuristic_bin_width(scores)
            n_bins = np.ceil(sp.ptp(scores)/width_score).astype(int)
            print("n_bins argument not given. "
                  "Automatically setting n_bins = {}.".format(n_bins))

        # approx distribution of scores -- called F
        counts, score_bins = np.histogram(scores, bins=n_bins)

        # generate cumulative dist of sqrt(F)
        sqrt_counts = np.sqrt(counts)
        csf = np.cumsum(sqrt_counts)

        if goal_num_strata == 'auto':
            # choose heuristically
            width_csf = _heuristic_bin_width(csf)
            goal_num_strata = np.ceil(sp.ptp(csf)/width_csf).astype(int)
            print("Automatically setting goal_num_strata = {}.".format(goal_num_strata))
        elif method == 'cum_sqrt_F':
            width_csf = csf[-1]/goal_num_strata

    # goal_num_strata is now guaranteed to have a valid integer value

    if method == 'equal_size':
        sorted_ids = scores.argsort()
        n_items = len(sorted_ids)
        quotient = n_items // goal_num_strata
        remainder = n_items % goal_num_strata
        allocations = []

        st_pops = [quotient for i in range(goal_num_strata)]
        for i in range(remainder):
            st_pops[i] += 1

        j = 0
        for k,nk in enumerate(st_pops):
            start = j
            end = j + nk
            allocations.append(np.array(sorted_ids[start:end]))
            j = end

    if method == 'cum_sqrt_F':
        if goal_num_strata > n_bins:
            warnings.warn("goal_num_strata > n_bins. "
                          "Consider increasing n_bins.")
        # calculate roughly equal bins on cum sqrt(F) scale
        csf_bins = [x * width_csf for x in np.arange(goal_num_strata + 1)]

        # map cum sqrt(F) bins to score bins
        j = 0
        new_bins = []
        for (idx,value) in enumerate(csf):
            if j == (len(csf_bins) - 1) or idx == (len(csf) - 1):
                new_bins.append(score_bins[-1])
                break
            if value >= csf_bins[j]:
                new_bins.append(score_bins[idx])
                j += 1
        new_bins[0] -= 0.01
        new_bins[-1] += 0.01

        # calculate stratum ids
        stratum_ids = np.digitize(scores, bins=new_bins, right=True) - 1

        # remove stratum ids with population zero
        nonempty_ids = np.unique(stratum_ids)
        num_strata = len(nonempty_ids)
        indices = np.arange(num_strata)
        stratum_ids = np.digitize(stratum_ids, nonempty_ids, right=True)

        allocations = []
        for k in indices:
            allocations.append(np.where(stratum_ids == k)[0])

        if len(allocations) < goal_num_strata:
            warnings.warn("Failed to create {} strata".format(goal_num_strata))

    return allocations

class Strata:
    """
    Represents a collection of strata, which contain data points. The data
    points are referenced by a unique index. Information about the data points
    must be stored elsewhere.

    Input
    -----
    allocations : 'by_features' or 'by_scores' or list of integer numpy.ndarrays
        Specifies how to allocate datapoints to strata. The pre-defined methods
        ('by_features' or  'by_scores') require additional arguments to be
        specified. Both methods require the `num_strata` argument, and
        'by_features' additionally requires the `features` argument.
        To use pre-determined stratum allocations, pass a list of numpy arrays.
        Each array in the list corresponds to a stratum, and should contain
        a set of datapoint indices.

    scores : float numpy.ndarray of length num_items
        Array of real-valued scores for each datapoint. The location in the
        array is the datapoint index.

    score_threshold : float

    calibrated_score : boolean
        whether the scores given are calibrated probabilities

    preds : integer numpy.ndarray of length num_items, optional, default None
        Array of (0/1) label predictions for each datapoint. The location in
        the array is the datapoint index. If not given, predictions are
        calculated using preds = (scores >= score_threshold) * 1

    splitting : bool, optional, default False
        Experimental option, which splits strata when they become inhomogeneous

    num_strata : integer, optional, default None
        Number of strata. Only required if using a pre-defined method for
        allocations

    features : float (num_items x num_fts) numpy.ndarray, optional, default None
        Feature matrix where rows correspond to data points and columns
        correspond to features. Only required if using 'by_features' method for
        allocations.

    Properties
    ----------
    num_strata : integer
        Number of strata

    num_items : integer
        Number of datapoints allocated to the strata

    indices : numpy.ndarray of length num_strata
        Array of stratum indices

    populations : numpy.ndarray of length num_strata
        Array specifying how many datapoints are allocated to each stratum

    weights : numpy.ndarray of length num_strata
        Array specifying the weight of each stratum (population/num_items)

    labels : list of integer numpy.ndarrays
        Has the same structure as `allocations`. Each element of the list
        contains an array of true labels for a particular stratum.
    """
    def __init__(self, allocations):
        # TODO Allow to generate allocations by passing scores etc.
        # TODO Check that input is valid

        self.allocations = allocations

        # Calculate population for each stratum
        self.populations = np.array([len(ids) for ids in self.allocations])

        # Total number of datapoints
        self.num_items = np.sum(self.populations)

        # Number of strata
        self.num_strata = len(self.allocations)

        # Calculate weights
        self.weights = self.populations/self.num_items

        # Stratum indices
        self.indices = np.arange(self.num_strata, dtype=int)

        # Keep a record of which items have been sampled
        self.sampled = [np.repeat(False, x) for x in self.populations]

    def _sample_stratum(self, pmf=None, replace=True):
        """
        Sample from the stratum indices

        Input
        -----
        pmf : float numpy.ndarray of length num_strata, optional
            probability distribution to use when sampling from the strata. If
            None, use the stratum weights.

        Output
        ------
        a randomly selected stratum index in the set {0, 1, ..., num_strata - 1}
        """
        if pmf is None:
            # Use weights
            pmf = self.weights

        if not replace:
            # Find strata which have been fully sampled (i.e. are now empty)
            empty = np.array([np.all(x) for x in self.sampled])
            pmf[empty] = 0
            pmf = pmf/np.sum(pmf)

        return np.random.choice(self.indices, p = pmf)

    def _sample_in_stratum(self, stratum_idx, replace = True):
        """
        Sample a datapoint uniformly from a stratum

        Input
        -----
        stratum_idx : integer
            stratum index to sample from

        replace : bool, optional, default True
            whether to sample with or without replacement

        Output
        ------
        a randomly selected location in the stratum
        """
        if replace:
            stratum_loc = np.random.choice(self.populations[stratum_idx])
        else:
            # Extract only the unsampled items
            stratum_locs = np.where(~self.sampled[stratum_idx])[0]
            stratum_loc = np.random.choice(stratum_locs)

        # Record that item has been sampled
        self.sampled[stratum_idx][stratum_loc] = True
        # Get generic location
        loc = self.allocations[stratum_idx][stratum_loc]
        return loc

    def sample(self, pmf=None, replace=True):
        """
        Sample a datapoint

        Input
        -----
        pmf : float numpy array of length K (number of strata), optional
            probability distribution to use when sampling from the strata. If
            None, use the stratum weights.

        Output
        ------
        stratum_idx : integer
            the randomly selected stratum index

        loc : integer
            the randomly selected location in the stratum
        """
        stratum_idx = self._sample_stratum(pmf, replace=replace)
        loc = self._sample_in_stratum(stratum_idx, replace=replace)
        return loc, stratum_idx

    def intra_mean(self, values):
        # TODO Check that quantity is valid
        return np.array([np.mean(values[x]) for x in self.allocations])

    def reset(self):
        self.sampled = [np.repeat(False, x) for x in self.populations]
