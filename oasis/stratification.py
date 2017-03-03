import numpy as np
from scipy.special import expit
from sklearn.cluster import KMeans
import warnings
import copy

def stratify_by_features(features, num_st, preds = None, n_init = 10,
                         n_jobs = 1):
    """
    Input
    -----
    features : float (num_pts x num_fts) numpy.ndarray
        Feature matrix where rows correspond to data points and columns
        correspond to features.

    num_st : integer
        Number of strata to create.

    preds : integer numpy.ndarray of length num_pts, optional, default None
        Array of (0/1) label predictions for each datapoint. The location in
        the array is the datapoint index. If present, the datapoints
        are separated into two groups -- predicted positives and predicted
        negatives. Then points are allocated to strata separately within
        each group.

    n_init : integer, optional, default 10
        number of times to run K-Means with different centroid seeds

    n_jobs : integer, optional, default 1
        number of jobs (i.e. number of CPUs) to use for the K-Means method

    Output
    ------
    allocations : list of integer numpy.ndarrays
        This object is used to specify how the datapoints are allocated amongst
        a collection of strata. Each element of the list represents a stratum
        and points to an array of indices that specify the datapoints contained
        within the stratum.
    """

    n_pts = features.shape[0]

    stratum_ids = np.empty(n_pts, dtype='int')

    if preds is not None:
        if type(preds) is not np.ndarray:
            raise TypeError("preds must be a numpy.ndarray")
        if len(preds) != n_pts:
            raise ValueError("length of preds does not match shape of features")

        pos_pts = (preds == 1)
        n_pos_pts = np.sum(pos_pts)
        n_pos_st = np.ceil(n_pos_pts/n_pts * num_st).astype(int)
        n_neg_st = num_st - n_pos_st

        # Cluster positive pts
        km = KMeans(n_clusters=n_pos_st, n_init=n_init, n_jobs=n_jobs)
        stratum_ids[pos_pts] = km.fit_predict(X=features[pos_pts,:])

        # Cluster negative pts
        km = KMeans(n_clusters=n_neg_st, n_init=n_init, n_jobs=n_jobs)
        stratum_ids[~pos_pts] = km.fit_predict(X=features[~pos_pts,:]) + n_pos_st
    else:
        # Cluster all pts together
        km = KMeans(n_clusters=num_st, n_init=n_init, n_jobs=n_jobs)
        stratum_ids = km.fit_predict(X=features)

    # Convert to stratum allocations
    allocations = []
    for k in range(num_st):
        allocations.append(np.where(stratum_ids == k)[0])

    return allocations

def stratify_by_scores(scores, goal_num_st, method = 'equal_size',
                       preds = None):
    """
    Input
    -----
    scores : float numpy array of length n
        array containing scores for each data point.

    goal_num_st : integer
        number of strata to create. For some methods, this number is
        a goal -- it is not guaranteed. The actual number of strata created is
        returned as output.

    method : 'cum_sqrt_F' or 'equal_size', optional, default 'equal_size'

    preds : integer numpy array of length n, optional, default None
        array of predictions (0/1) for each data point. If present, the points
        are separated into two groups -- predicted positives and predicted
        negatives. Then points are allocated to strata separately within
        each group. (Note: not implemented for `cum_sqrt_F` yet)

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
    
    def alloc_equally(ids, n_st):
        n_pts = len(ids)
        quotient = n_pts // n_st
        remainder = n_pts % n_st
        allocations = []

        st_pops = [quotient for i in range(n_st)]
        for i in range(remainder):
            st_pops[i] += 1

        j = 0
        for k,nk in enumerate(st_pops):
            start = j
            end = j + nk
            allocations.append(np.array(ids[start:end]))
            j = end
        return allocations

    if method == 'equal_size':
        if preds is not None:
            pos_ids = np.where(preds == 1)[0]
            neg_ids = np.where(preds == 0)[0]
            n_pos_pts = len(pos_ids)
            n_pts = len(scores)
            n_pos_st = np.ceil(n_pos_pts/n_pts * goal_num_st).astype(int)
            n_neg_st = goal_num_st - n_pos_st

            sorted_pos_ids = pos_ids[scores[pos_ids].argsort()]
            sorted_neg_ids = neg_ids[scores[neg_ids].argsort()]

            allocations = alloc_equally(sorted_neg_ids, n_neg_st)
            allocations = allocations + alloc_equally(sorted_pos_ids, n_pos_st)
        else:
            sorted_ids = scores.argsort()
            allocations = alloc_equally(sorted_ids, goal_num_st)
    
    def cum_sqrt_F(ids, scores, n_bins, goal_num_st):
        # approx distribution of scores -- called F
        counts, score_bins = np.histogram(scores, bins=n_bins)
        
        # generate cumulative dist of sqrt(F)
        sqrt_counts = np.sqrt(counts)
        csf = np.cumsum(sqrt_counts)
        
        # equal bin width on cum sqrt(F) scale
        width_csf = csf[-1]/goal_num_st
        
        # calculate roughly equal bins on cum sqrt(F) scale
        csf_bins = [x * width_csf for x in np.arange(goal_num_st + 1)]
        
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
        stratum_ids = np.digitize(scores, bins = new_bins, right = True) - 1

        # remove stratum ids with population zero
        nonempty_ids = np.unique(stratum_ids)
        num_st = len(nonempty_ids)
        indices = np.arange(num_st)
        stratum_ids = np.digitize(stratum_ids, nonempty_ids, right=True)
        
        allocations = []
        for k in indices:
            allocations.append(ids[np.where(stratum_ids == k)[0]])
        
        return allocations
    
    def cum_sqrt_F_choose_n_bins(ids, scores, goal_num_st):
        n_bins = 10*goal_num_st
        num_st = 0
        n_iter = 10
        for i in range(n_iter):
            allocations = cum_sqrt_F(ids, scores, n_bins, goal_num_st)
            if num_st < goal_num_st:
                n_bins = 2*n_bins
                continue
            else:
                break
        
        return allocations
    
    # FIXME consider removing this option or implementing split_threshold
    if method == 'cum_sqrt_F':
        n_pts = len(scores)
        if preds is not None:
            pos_ids = np.where(preds == 1)[0]
            neg_ids = np.where(preds == 0)[0]
            n_pos_pts = len(pos_ids)
            n_pos_st = np.ceil(n_pos_pts/n_pts * goal_num_st).astype(int)
            n_neg_st = goal_num_st - n_pos_st
            
            allocations = cum_sqrt_F_choose_n_bins(neg_ids, scores[neg_ids], n_neg_st)
            allocations = allocations + \
                            cum_sqrt_F_choose_n_bins(pos_ids, scores[pos_ids], n_pos_st)
        else:
            allocations = cum_sqrt_F_choose_n_bins(np.arange(n_pts), scores, goal_num_st)
        
        if len(allocations) < goal_num_st:
            warnings.warn("Failed to create {} strata".format(goal_num_st))

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
        specified. Both methods require the `num_st` argument, and
        'by_features' additionally requires the `features` argument.
        To use pre-determined stratum allocations, pass a list of numpy arrays.
        Each array in the list corresponds to a stratum, and should contain
        a set of datapoint indices.

    scores : float numpy.ndarray of length num_pts
        Array of real-valued scores for each datapoint. The location in the
        array is the datapoint index.

    score_threshold : float

    calibrated_score : boolean
        whether the scores given are calibrated probabilities

    preds : integer numpy.ndarray of length num_pts, optional, default None
        Array of (0/1) label predictions for each datapoint. The location in
        the array is the datapoint index. If not given, predictions are
        calculated using preds = (scores >= score_threshold) * 1

    splitting : bool, optional, default False
        Experimental option, which splits strata when they become inhomogeneous

    num_st : integer, optional, default None
        Number of strata. Only required if using a pre-defined method for
        allocations

    features : float (num_pts x num_fts) numpy.ndarray, optional, default None
        Feature matrix where rows correspond to data points and columns
        correspond to features. Only required if using 'by_features' method for
        allocations.

    Properties
    ----------
    num_st : integer
        Number of strata

    num_pts : integer
        Number of datapoints allocated to the strata

    indices : numpy.ndarray of length num_st
        Array of stratum indices

    populations : numpy.ndarray of length num_st
        Array specifying how many datapoints are allocated to each stratum

    weights : numpy.ndarray of length num_st
        Array specifying the weight of each stratum (population/num_pts)

    labels : list of integer numpy.ndarrays
        Has the same structure as `allocations`. Each element of the list
        contains an array of true labels for a particular stratum.
    """
    def __init__(self, allocations, scores, score_threshold,
                 calibrated_score, preds = None, features = None, num_st = None,
                 splitting = False):
        # Generate stratum allocations
        if allocations == 'by_features':
            if features is None:
                raise ValueError("`features` argument cannot be None")
            if num_st is None:
                raise ValueError("`num_st` argument cannot be None")
            self.allocations = stratify_by_features(features, num_st,
                                    preds=preds, n_init=10, n_jobs=1)
        elif allocations == 'by_scores':
            if num_st is None:
                raise ValueError("`num_st` argument cannot be None")
            self.allocations = stratify_by_scores(scores, num_st,
                                    preds=preds, method='equal_size')
        else:
            self.allocations = allocations

        # Calculate population for each stratum
        self.populations = np.array([len(ids) for ids in self.allocations])

        # Total number of datapoints
        self.num_pts = np.sum(self.populations)

        # Number of strata
        self.num_st = len(self.allocations)

        # Calculate weights
        self.weights = self.populations/self.num_pts

        # Stratum indices
        self.indices = np.arange(self.num_st, dtype=int)

        # Store scores as list of arrays
        self.scores = []
        for k in self.indices:
            self.scores.append(scores[self.allocations[k]])

        # Store mean score per stratum
        self.mean_score = np.array([np.mean(x) for x in self.scores])

        # Store predictions as list of arrays
        if preds is None:
            preds = (scores >= score_threshold)*1
        self.preds = []
        for k in self.indices:
            self.preds.append(preds[self.allocations[k]])

        # Store mean prediction per stratum
        self.mean_pred = np.array([np.mean(x) for x in self.preds])

        # Sort points within strata in ascending order of score
        for k in self.indices:
            sorted_id = np.argsort(self.scores[k])
            self.scores[k] = self.scores[k][sorted_id]
            self.allocations[k] = self.allocations[k][sorted_id]
            self.preds[k] = self.preds[k][sorted_id]

        self.score_threshold = score_threshold
        self.calibrated_score = calibrated_score
        self.splitting = splitting

        # Allocate arrays to store true labels
        self.labels = [np.repeat(np.nan, x) for x in self.populations]

    def _add_stratum(self):
        """
        Add a new empty stratum
        """
        new_stratum_id = np.max(self.indices) + 1
        self.indices = np.append(self.indices, new_stratum_id)
        self.num_st += 1
        self.allocations.append(np.array([], dtype='int'))
        self.populations = np.append(self.populations, 0)
        self.scores.append(np.array([], dtype='float'))
        self.mean_score = np.append(self.mean_score, 0)
        self.preds.append(np.array([], dtype='float'))
        self.mean_pred = np.append(self.mean_pred, 0)
        self.weights = np.append(self.weights, 0)
        self.labels.append(np.array([], dtype='int'))

        return new_stratum_id

    def _sample_stratum(self, prob_dist=None, replace=True):
        """
        Sample from the stratum indices

        Input
        -----
        prob_dist : float numpy.ndarray of length num_st, optional
            probability distribution to use when sampling from the strata. If
            None, use the stratum weights.

        Output
        ------
        a randomly selected stratum index in the set {0, 1, ..., num_st - 1}
        """
        if prob_dist is None:
            # Use weights
            prob_dist = self.weights

        if not replace:
            nonempty = np.array([np.isnan(np.sum(x)) for x in self.labels])
            prob_dist[~nonempty] = 0
            prob_dist = prob_dist/np.sum(prob_dist)

        return np.random.choice(self.indices, p = prob_dist)

    def _sample_in_stratum(self, stratum_id, replace = True):
        """
        Sample a datapoint uniformly from a stratum

        Input
        -----
        stratum_id : integer
            stratum index to sample from

        replace : bool, optional, default True
            whether to sample with or without replacement

        Output
        ------
        a randomly selected location in the stratum
        """

        if replace:
            points = self.populations[stratum_id]
        else:
            # Sample only from points whose labels haven't been seen
            points = np.where(np.isnan(self.labels[stratum_id]))[0]

        loc = np.random.choice(points)

        return loc

    def sample(self, prob_dist=None, replace=True):
        """
        Sample a datapoint

        Input
        -----
        prob_dist : float numpy array of length K (number of strata), optional
            probability distribution to use when sampling from the strata. If
            None, use the stratum weights.

        Output
        ------
        stratum_id : integer
            the randomly selected stratum index

        loc : integer
            the randomly selected location in the stratum
        """
        stratum_id = self._sample_stratum(prob_dist, replace=replace)

        loc = self._sample_in_stratum(stratum_id, replace=replace)

        return stratum_id, loc

    def update_label(self, stratum_id, loc, label):
        """
        Use this method to update the label in a stratum at a particular
        location
        """
        self.labels[stratum_id][loc] = label

    def split(self, st_id, loc, label):
        """
        Splits the stratum with index `st_id` after an anomalous point is
        sampled at location `loc` with label `label`.
        """
        # Check whether it is possible to split the stratum
        if self.populations[st_id] <= 1:
            warnings.warn("Cannot split stratum {}".format(st_id))
            return

        # Create a new empty stratum
        new_st_id = self._add_stratum()

        # Find locations of labelled points
        unlabelled = np.isnan(self.labels[st_id])
        # Exclude the recently sampled point from the labelled set, since
        # it will not remain in the old stratum
        labelled = ~unlabelled
        labelled[loc] = False
        labelled_loc = np.where(labelled)[0]
        unlabelled_loc = np.where(unlabelled)[0]

        n = self.populations[st_id]
        n_unlabelled = len(unlabelled_loc)
        n_old = np.ceil(n_unlabelled/2).astype(int)
        n_new = n_unlabelled - n_old

        # Split using scores -- try to keep points with lower scores with
        # labelled zeros, and points with higher scores with labelled ones.
        if label == 1:
            loc_old = np.append(labelled_loc, unlabelled_loc[0:n_old])
            loc_new = np.append(loc, unlabelled_loc[n_old:n_unlabelled])
        else:
            # label == 0
            loc_old = np.append(labelled_loc, unlabelled_loc[n_new:n_unlabelled])
            loc_new = np.append(loc, unlabelled_loc[0:n_new])

        # Preserve ordering (sorted by score in ascending order)
        loc_old = np.sort(loc_old)
        loc_new = np.sort(loc_new)

        self.allocations[new_st_id] = self.allocations[st_id][loc_new]
        self.populations[new_st_id] = n_new + 1
        self.weights[new_st_id] = self.populations[new_st_id]/self.num_pts
        self.labels[new_st_id] = self.labels[st_id][loc_new]
        self.scores[new_st_id] = self.scores[st_id][loc_new]
        self.preds[new_st_id] = self.preds[st_id][loc_new]
        self.mean_score[new_st_id] = np.mean(self.scores[new_st_id])
        self.mean_pred[new_st_id] = np.mean(self.preds[new_st_id])

        self.allocations[st_id] = self.allocations[st_id][loc_old]
        self.populations[st_id] = n - n_unlabelled - 1 + n_old
        self.weights[st_id] = self.populations[st_id]/self.num_pts
        self.labels[st_id] = self.labels[st_id][loc_old]
        self.scores[st_id] = self.scores[st_id][loc_old]
        self.preds[st_id] = self.preds[st_id][loc_old]
        self.mean_score[st_id] = np.mean(self.scores[st_id])
        self.mean_pred[st_id] = np.mean(self.preds[st_id])
