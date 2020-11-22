import numpy as np
from collections import Counter


def bin_width_using_freedman_diaconis_rule(obs):
    IQR = np.percentile(obs, 75) - np.percentile(obs, 25)
    N = len(obs)
    return 2 * IQR * N ** (-1 / 3)


def stratify_by_equal_size_method(scores, goal_n_strata=None):
    if not goal_n_strata:
        strata_width = bin_width_using_freedman_diaconis_rule(scores)
        goal_n_strata = np.ceil(np.ptp(scores) / strata_width).astype(np.int)
    n_items = len(scores)
    sorted_ids = scores.argsort()
    quotient = n_items // goal_n_strata
    remainder = n_items % goal_n_strata
    print("q and r", quotient, remainder)
    allocations = np.empty(n_items, dtype='int')
    st_pops = (np.repeat(quotient, goal_n_strata) + np.concatenate((np.ones(remainder), np.zeros(goal_n_strata-remainder))))\
        .cumsum().astype(int)

    for k, (start, end) in enumerate(zip(np.concatenate((np.zeros(1), st_pops)).astype(int), st_pops)):
        print(start, end)
        allocations[sorted_ids[start:end]] = k

    #print(set(Counter(allocations).values()))
    return allocations


def stratify_by_cum_sqrt_f_method(scores):
    score_width = bin_width_using_freedman_diaconis_rule(scores)
    n_bins = np.ceil(np.ptp(scores) / score_width).astype(int)
    counts, score_bins = np.histogram(scores, bins=n_bins)
    csf = np.sqrt(counts).cumsum() # cum sqrt(F)
    strata_width = bin_width_using_freedman_diaconis_rule(csf)
    new_bins = []
    j = 0
    for x, sb in zip(csf, score_bins):
        if x >= strata_width * j:
            new_bins.append(sb)
            j += 1

    new_bins.append(score_bins[-1])
    # add margin
    new_bins[0] -= 0.01
    new_bins[-1] += 0.01

    return np.digitize(scores, bins=new_bins, right=True) - 1


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
            probability distribution to use when sampling from the    strata. If
            not given, use the stratum weights.

        replace : bool, optional, default True
            whether to sample with replacement

        Returns
        -------
        int
            a randomly selected stratum index
        """
        pmf = pmf or self.weights_
        if replace:
            # Find strata which have been fully sampled (i.e. are now empty)
            return np.random.choice(self.indices_, p=pmf)

        empty = (self._n_sampled >= self.sizes_)
        if np.any(empty):
            pmf = pmf[~empty]
            if np.sum(pmf) == 0:
                raise RuntimeError("all datapoints have been sampled")
            pmf /= np.sum(pmf)

        return np.random.choice(self.indices_[~empty], p=pmf)


    def _sample_in_stratum(self, stratum_idx, replace=True):
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
            stratum_locs, = np.where(~self._sampled[stratum_idx])
            stratum_loc = np.random.choice(stratum_locs)

        # Record that item has been sampled
        self._sampled[stratum_idx][stratum_loc] = True
        self._n_sampled[stratum_idx] += 1
        # Get generic location
        loc = self.allocations_[stratum_idx][stratum_loc]
        return loc

    def sample(self, replace=True):
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
        stratum_idx = self._sample_stratum(replace=replace)
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
        return np.array([np.mean(values[x]) for x in self.allocations_])

    def reset(self):
        """Reset the instance to begin sampling from scratch"""
        self._sampled = [np.repeat(False, x) for x in self.sizes_]
        self._n_sampled = np.zeros(self.n_strata_, dtype=int)
