import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import warnings
import copy

def stratify_by_features(features, n_strata, **kwargs):
    """Stratify by clustering the items in feature space

    Parameters
    ----------
    features : array-like, shape=(n_items,n_features)
        feature matrix for the pool, where rows correspond to items and columns
        correspond to features.

    n_strata : int
        number of strata to create.

    **kwargs :
        passed to sklearn.cluster.KMeans

    Returns
    -------
    Strata instance
    """
    n_items = features.shape[0]
    km = KMeans(n_clusters=n_strata, **kwargs)
    allocations = km.fit_predict(X=features)
    return Strata(allocations)

def _heuristic_bin_width(obs):
    """Optimal histogram bin width based on the Freedman-Diaconis rule"""
    IQR = sp.percentile(obs, 75) - sp.percentile(obs, 25)
    N = len(obs)
    return 2*IQR*N**(-1/3)

def stratify_by_scores(scores, goal_n_strata='auto', method='cum_sqrt_F',
                       n_bins = 'auto'):
    """Stratify by binning the items based on their scores

    Parameters
    ----------
    scores : array-like, shape=(n_items,)
        ordered array of scores which quantify the classifier confidence for
        the items in the pool. High scores indicate a high confidence that
        the true label is a "1" (and vice versa for label "0").

    goal_n_strata : int or 'auto', optional, default 'auto'
        desired number of strata. If set to 'auto', the number is selected
        using the Freedman-Diaconis rule. Note that for the 'cum_sqrt_F' method
        this number is a goal -- the actual number of strata created may be
        less than the goal.

    method : {'cum_sqrt_F' or 'equal_size'}, optional, default 'cum_sqrt_F'
        stratification method to use. 'equal_size' aims to create s

    Other Parameters
    ----------------
    n_bins : int or 'auto', optional, default 'auto'
        specify the number of bins to use when estimating the distribution of
        the score function. This is used when ``goal_n_strata = 'auto'``
        and/or when ``method = 'cum_sqrt_F'``. If set to 'auto', the number is
        selected using the Freedman-Diaconis rule.

    Returns
    -------
    Strata instance
    """

    available_methods = ['equal_size', 'cum_sqrt_F']
    if method not in available_methods:
        raise ValueError("method argument is invalid")

    if (method == 'cum_sqrt_F') or (goal_n_strata == 'auto'):
        # computation below is needed for cum_sqrt_F method OR if we need to
        # determine the number of strata for equal_size method automatically
        if n_bins == 'auto':
            # choose n_bins heuristically
            width_score = _heuristic_bin_width(scores)
            n_bins = np.ceil(sp.ptp(scores)/width_score).astype(int)
            print("Automatically setting n_bins = {}.".format(n_bins))

        # approx distribution of scores -- called F
        counts, score_bins = np.histogram(scores, bins=n_bins)

        # generate cumulative dist of sqrt(F)
        sqrt_counts = np.sqrt(counts)
        csf = np.cumsum(sqrt_counts)

        if goal_n_strata == 'auto':
            # choose heuristically
            width_csf = _heuristic_bin_width(csf)
            goal_n_strata = np.ceil(sp.ptp(csf)/width_csf).astype(int)
            print("Automatically setting goal_n_strata = {}.".format(goal_n_strata))
        elif method == 'cum_sqrt_F':
            width_csf = csf[-1]/goal_n_strata

    # goal_n_strata is now guaranteed to have a valid integer value

    if method == 'equal_size':
        sorted_ids = scores.argsort()
        n_items = len(sorted_ids)
        quotient = n_items // goal_n_strata
        remainder = n_items % goal_n_strata
        allocations = np.empty(n_items, dtype='int')

        st_pops = [quotient for i in range(goal_n_strata)]
        for i in range(remainder):
            st_pops[i] += 1

        j = 0
        for k,nk in enumerate(st_pops):
            start = j
            end = j + nk
            allocations[sorted_ids[start:end]] = k
            j = end

    if method == 'cum_sqrt_F':
        if goal_n_strata > n_bins:
            warnings.warn("goal_n_strata > n_bins. "
                          "Consider increasing n_bins.")
        # calculate roughly equal bins on cum sqrt(F) scale
        csf_bins = [x * width_csf for x in np.arange(goal_n_strata + 1)]

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

        # bin scores based on new_bins
        allocations = np.digitize(scores, bins=new_bins, right=True) - 1

        # remove empty strata
        nonempty_ids = np.unique(allocations)
        n_strata = len(nonempty_ids)
        indices = np.arange(n_strata)
        allocations = np.digitize(allocations, nonempty_ids, right=True)

        if n_strata < goal_n_strata:
            warnings.warn("Failed to create {} strata".format(goal_n_strata))

    return Strata(allocations)

def auto_stratify(scores, **kwargs):
    """Generate Strata instance automatically

    Parameters
    ----------
    scores : array-like, shape=(n_items,)
        ordered array of scores which quantify the classifier confidence for
        the items in the pool. High scores indicate a high confidence that
        the true label is a "1" (and vice versa for label "0").

    **kwargs :
        optional keyword arguments. May include 'stratification_method',
        'stratification_n_strata', 'stratification_n_bins'.

    Returns
    -------
    Strata instance
    """
    if 'stratification_method' in kwargs:
        method = kwargs['stratification_method']
    else:
        method = 'cum_sqrt_F'
    if 'stratification_n_strata' in kwargs:
        n_strata = kwargs['stratification_n_strata']
    else:
        n_strata = 'auto'
    if 'stratification_n_bins' in kwargs:
        n_bins = kwargs['stratification_n_bins']
        strata = stratify_by_scores(scores, n_strata, method = method, \
                                         n_bins = n_bins)
    else:
        strata = stratify_by_scores(scores, n_strata, method = method)
    return strata

class Strata:
    """Represents a collection of strata and facilitates sampling from them

    This class takes an array of prescribed stratum allocations for a finite
    pool and stores the information in a form that is convenient for
    sampling. The items in the pool are referred to uniquely by their location
    in the input array. An item may be sampled from the strata (according to an
    arbitrary distribution over the strata) using the `sample` method.

    Parameters
    ----------
    allocations : array-like, shape=(n_items,)
        ordered array of ints or strs which specifies the name/identifier of
        the allocated stratum for each item in the pool.

    Attributes
    ----------
    allocations_ : list of numpy.ndarrays, length n_strata
        represents the items contained within each stratum using a list of
        arrays. Each array in the list refers to a particular stratum, and
        stores the items contained within that stratum. Items are referred to
        by their location in the input array.

    n_strata_ : int
        number of strata

    n_items_ : int
        number of items in the pool (i.e. in all of the strata)

    names_ : numpy.ndarray, shape=(n_strata,)
        array containing names/identifiers for each stratum

    indices_ : numpy.ndarray, shape=(n_strata,)
        array containing unique indices for each stratum

    sizes_ : numpy.ndarray, shape=(n_strata,)
        array specifying how many items are contained with each stratum

    weights_ : numpy.ndarray, shape=(n_strata,)
        array specifying the stratum weights (sizes/n_items)
    """
    def __init__(self, allocations):
        # TODO Check that input is valid

        # Names of strata (could be ints or strings for example)
        self.names_ = np.unique(allocations)

        # Number of strata
        self.n_strata_ = len(self.names_)

        # Size of pool
        self.n_items_ = len(allocations)

        self.allocations_ = []
        for name in self.names_:
            self.allocations_.append(np.where(allocations == name)[0])

        # Calculate population for each stratum
        self.sizes_ = np.array([len(ids) for ids in self.allocations_])

        # Calculate weights
        self.weights_ = self.sizes_/self.n_items_

        # Stratum indices
        self.indices_ = np.arange(self.n_strata_, dtype=int)

        # Keep a record of which items have been sampled
        self._sampled = [np.repeat(False, x) for x in self.sizes_]

        # Keep a record of how many items have been sampled
        self._n_sampled = np.zeros(self.n_strata_, dtype=int)

    def _sample_stratum(self, pmf=None, replace=True):
        """Sample a stratum

        Parameters
        ----------
        pmf : array-like, shape=(n_strata,), optional, default None
            probability distribution to use when sampling from the strata. If
            not given, use the stratum weights.

        replace : bool, optional, default True
            whether to sample with replacement

        Returns
        -------
        int
            a randomly selected stratum index
        """
        if pmf is None:
            # Use weights
            pmf = self.weights_

        if not replace:
            # Find strata which have been fully sampled (i.e. are now empty)
            empty = (self._n_sampled >= self.sizes_)
            if np.any(empty):
                pmf = copy.copy(pmf)
                pmf[empty] = 0
                if np.sum(pmf) == 0:
                    raise(RuntimeError)
                pmf /= np.sum(pmf)

        return np.random.choice(self.indices_, p = pmf)

    def _sample_in_stratum(self, stratum_idx, replace = True):
        """Sample an item uniformly from a stratum

        Parameters
        ----------
        stratum_idx : int
            stratum index to sample from

        replace : bool, optional, default True
            whether to sample with replacement

        Returns
        -------
        int
            location of the randomly selected item in the original input array
        """
        if replace:
            stratum_loc = np.random.choice(self.sizes_[stratum_idx])
        else:
            # Extract only the unsampled items
            stratum_locs = np.where(~self._sampled[stratum_idx])[0]
            stratum_loc = np.random.choice(stratum_locs)

        # Record that item has been sampled
        self._sampled[stratum_idx][stratum_loc] = True
        self._n_sampled[stratum_idx] += 1
        # Get generic location
        loc = self.allocations_[stratum_idx][stratum_loc]
        return loc

    def sample(self, pmf=None, replace=True):
        """Sample an item from the strata

        Parameters
        ----------
        pmf : array-like, shape=(n_strata,), optional, default None
            probability distribution to use when sampling from the strata. If
            not given, use the stratum weights.

        replace : bool, optional, default True
            whether to sample with replacement

        Returns
        -------
        loc : int
            location of the randomly selected item in the original input array

        stratum_idx : int
            the stratum index that was sampled from
        """
        stratum_idx = self._sample_stratum(pmf, replace=replace)
        loc = self._sample_in_stratum(stratum_idx, replace=replace)
        return loc, stratum_idx

    def intra_mean(self, values):
        """Calculate the mean of a quantity within strata

        Parameters
        ----------
        values : array-like, shape=(n_items,n_class)
            array containing the values of the quantity for each item in the
            pool

        Returns
        -------
        numpy.ndarray, shape=(n_strata,n_class)
            array containing the mean value of the quantity within each stratum
        """
        # TODO Check that quantity is valid
        if values.ndim > 1:
            return np.array([np.mean(values[x,:], axis=0) for x in self.allocations_])
        else:
            return np.array([np.mean(values[x]) for x in self.allocations_])

    def reset(self):
        """Reset the instance to begin sampling from scratch"""
        self._sampled = [np.repeat(False, x) for x in self.sizes_]
        self._n_sampled = np.zeros(self.n_strata_, dtype=int)
