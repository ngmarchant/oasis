import numpy as np
import warnings

class Passive:
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
    def __init__(self, labels, preds, alpha = 0.5, max_iter = None, 
                 replace = True, debug = False):
        self.debug = debug
        self.labels = labels
        self.alpha = alpha
        self.t = 0
        self.preds = preds
        self._max_iter = max_iter
        self._num_pts = len(labels)
        self.replace = replace

        if self._max_iter is None:
            self._max_iter = self._num_pts
            
        # Terms used to calculate F-measure (we update them iteratively)
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0
        
        # Array to record history of F-measure estimates
        self.F = np.repeat(np.nan, self._max_iter)
        
        # Array to record whether oracle was queried at each iteration (don't
        # need this if sampling without replacement)
        if self.replace:
            self.queried_oracle = np.repeat(False, self._max_iter)
            
        # Array to record whether label for each point has been seen
        self.seen_labels = np.repeat(False, self._num_pts)
        
    def _update_F_terms(self, y, yhat):
        """
        Iteratively update the terms that are used to calculate the F-measure 
        after a new point is sampled with label `y` and prediction `yhat`.
        """
        
        if y == 1 and yhat == 1:
            # Point is true positive
            self._TP_term = self._TP_term + 1
            self._PP_term = self._PP_term + 1
            self._P_term = self._P_term + 1
        elif yhat == 1:
            # Point is false positive
            self._PP_term = self._PP_term + 1
        elif y == 1:
            # Point is false negative
            self._P_term = self._P_term + 1
    
    def _calc_F(self):
        """
        Returns the latest estimate of the F-measure.
        """
        
        num = self._TP_term
        den = (self.alpha * self._PP_term + (1 - self.alpha) * self._P_term)
        
        if den == 0:
            return np.nan
        else:
            return num/den
    
    def _sample_pt(self):
        """
        Samples a point according to the "optimal" instrumental distribution
        with replacement.
        """
        
        if self.replace:
            # Can sample from any of the points
            sample_id = np.random.choice(self._num_pts)
        else:
            # Can only sample from points that have not been seen
            # Find ids that haven't been seen yet
            not_seen_ids = np.where(~self.seen_labels)[0]
            sample_id = np.random.choice(not_seen_ids)
        
        return sample_id
    
    def _query_label(self, t, sample_idx):
        """
        Queries the label for the point with id `sample_idx`. Also keeps
        a record of which labels have been seen before and which labels have
        been queried from the oracle (for the first time).
        
        Returns the ground truth label `0` or `1`.
        """ 
               
        if self.seen_labels[sample_idx]:
            # Have already seen label
            y = self.labels[sample_idx]
            if self.replace:
                self.queried_oracle[t] = False
            return y
               
        # Otherwise need to query oracle
        y = self.labels[sample_idx]
        self.seen_labels[sample_idx] = True
        if self.replace:
            self.queried_oracle[t] = True
        return y
    
    def reset(self):
        """
        Resets the instance to begin sampling again
        """
        self.t = 0
           
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0
        
        self.F = np.repeat(np.nan, self._max_iter)
        
        if self.replace:
            self.queried_oracle = np.repeat(False, self._max_iter)
        self.seen_labels = np.repeat(False, self._num_pts)        
        
    def sample(self, n_iter):
        """
        Samples `n_iter` points
        """
        t_i = self.t
        t_f = n_iter + self.t
        
        assert t_f <= self.F.shape[0]
        
        if (not self.replace) and (t_f > self._num_pts):
            n_remaining = self._num_pts - t_i
            warnings.warn("Cannot sample {} labels without replacement when only {} unsampled labels remain. Setting n_iter = {}.".format(n_iter,n_remaining,n_remaining))
            t_f = self._num_pts
        
        for t in range(t_i, t_f):
           
            # Sample label
            sample_idx = self._sample_pt()
            
            # Query label
            y = self._query_label(t, sample_idx)
            
            if self.debug == True:
                print("Sampled label {} for point {}".format(y,sample_idx))
            
            self._update_F_terms(y, self.preds[sample_idx])
            self.F[t] = self._calc_F()
            
            self.t = self.t + 1
            
    def sample_until(self, n_goal):
        """
        Sample until `n_goal` labels are queried from the oracle
        """
        
        n_seen = np.sum(self.seen_labels)
        
        if n_seen >= n_goal:
            raise Exception("Sampling goal already met: n_goal <= n_seen = {}".format(n_seen))
        
        if n_goal > self._max_iter:
            raise ValueError("n_goal cannot be greater than max_iter".format(n_goal,self._max_iter))
        
        if (not self.replace) and n_goal > self._num_pts:
            warnings.warn("Cannot sample {} labels without replacement from a pool of size {}. Setting n_goal = {}.".format(n_goal,self._num_pts, self._num_pts))
            n_goal = self._num_pts

        while n_seen < n_goal:
            self.sample(1)
            if self.replace:
                seen = self.queried_oracle[self.t - 1]*1
            else:
                # Sampling without replacement, so must have seen a new label
                seen = 1
            n_seen = n_seen + seen
