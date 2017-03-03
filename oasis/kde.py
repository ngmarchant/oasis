import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
import copy

    
        
class KDE:
    """
    Input
    -----
    labels:   array of ground truth labels (0 or 1). The position in the array
              is assumed to be the identifier of the point.
    
    scores:   array of scores. The position in 
              the array is assumed to be the identifier of the point.

    alpha:    double on the unit interval. Indicates the weight to use for
              the F-measure. alpha = 0.5 is balanced, alpha = 1 corresponds to
              precision and alpha = 0 corresponds to recall. (default: 0.5)
    
    score_threshold : double on the unit interval. Indicates the threshold to use
              for classification. If score >= score_threshold, the point is classified as 1,
              otherwise, the point is classified as 0. (default: 0)
    
    max_iter: positive integer. Maximum number of labels to be drawn. Used to
              preallocate arrays. (default: None)
    
    epsilon:  double on the interval open interval (0, 1). Explore vs. exploit 
              parameter. The sampling is close to "optimal" for small epsilon 
              and close to passive for large epsilon. (default: 1e-3).
        
    debug:    boolean. If True, print verbose information. (default: False)
    """
    def __init__(self, features, scores, labels, alpha = 0.5, score_threshold = 0.0, 
                 prob_threshold = 0.5, max_iter = None, epsilon = 1e-3, debug = False, passive = False):
        self.debug = debug
        self.features = features
        self.labels = labels
        self.scores = scores
        self.predictions = (scores >= score_threshold)*1
        self.alpha = alpha
        self.score_threshold = score_threshold
        self.epsilon = epsilon
        self.t = 0
        self._max_iter = max_iter
        self._F_prob_est = None
        self._num_pts = len(scores)
        self.passive = passive
        self.prob_threshold = prob_threshold
        
        if self._max_iter is None:
            self._max_iter = self.strata.num_pts
            
        # Terms used to calculate F-measure (we update them iteratively)
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0
        
        # Array to record history of F-measure estimates
        self.F = np.repeat(np.nan, self._max_iter)
        # Array to record whether oracle was queried at each iteration
        self.queried_oracle = np.repeat(False, self._max_iter)
        # Array to record whether label at location 'idx' has been seen
        self.seen_labels = np.repeat(False, self._num_pts)
        
        self.pmf = np.empty(self._num_pts)
        self.density = np.empty(self._num_pts)
        self.density_1 = np.empty(self._num_pts)
        self.norm_const = 0
        self._calc_density()
        self.p_labels = (scores > score_threshold)*1
        self.p_probs = np.empty(self._num_pts)
        self._update_p_probs()
        
    def _calc_density(self, rtol = 1e-3, bandwidth = 0.5):
        # TODO find optimal bandwidth through cross-validation
        kde = KernelDensity(rtol = rtol, bandwidth = bandwidth)
        print("Running KDE. This may take some time.")
        kde.fit(self.features)
        self.density = np.exp(kde.score_samples(self.features) 
                                + np.log(self._num_pts))
        # Normalise
        self.norm_const = np.sum(self.density)
        self.density = self.density/self.norm_const
        print("KDE complete.")
    
    def _update_p_probs(self, rtol = 1e-3, bandwidth = 0.5):
    
        features = self.features
        norm_const = self.norm_const
        # TODO count numbers of 1's based on p_probs instead
        n_1 = np.count_nonzero(self.p_labels)
        p_ratio_1 = n_1/self._num_pts
        p_ratio_0 = 1 - p_ratio_1
        
        kde = KernelDensity(bandwidth = bandwidth, rtol=rtol)
        kde.fit(features[np.where(self.p_labels == 1)[0],:])
        self.density_1 = np.exp(kde.score_samples(features) + np.log(n_1))
        self.density_1 = self.density_1/self.norm_const
        density_1 = self.density_1
        density_0 = self.density - self.density_1
        self.p_probs = ( density_1 * p_ratio_1 / 
                        (density_0 * p_ratio_0 + density_1 * p_ratio_1) )
        
    def _update_F_terms(self, y, yhat, w):
        """
        Iteratively update the terms that are used to calculate the F-measure 
        after a new point is sampled with weight `w`, label `y` and prediction 
        `yhat`.
        """
        if y == 1 and yhat == 1:
            # Point is true positive
            self._TP_term = self._TP_term + w
            self._PP_term = self._PP_term + w
            self._P_term = self._P_term + w
        elif yhat == 1:
            # Point is false positive
            self._PP_term = self._PP_term + w
        elif y == 1:
            # Point is false negative
            self._P_term = self._P_term + w
            
    def _update_F(self):
        """
        Records the latest estimate of the F-measure
        """
        
        t = self.t
            
        num = self._TP_term
        den = (self.alpha * self._PP_term + (1 - self.alpha) * self._P_term)
        
        if den == 0:
            self.F[t] = np.nan
        else:
            self.F[t] = num/den
   
    def _score_to_prediction(self, score):
        """
        Returns the predition (0 or 1) for a given score
        """
        if score >= self.score_threshold:
            return 1
        else:
            return 0
            
    def _sample_pt(self):
        """
        Samples a point according to the "optimal" instrumental distribution
        with replacement.
        """
        return np.random.choice(self._num_pts, p = self.pmf)
            
    def _query_label(self, sample_id):
        """
        Queries the label for the point with id `sample_id`. Also keeps a 
        record of which labels have been seen before and which labels have
        been queried from the oracle (for the first time).
        
        Returns the ground truth label `0` or `1`.
        """ 
        
        t = self.t
               
        if self.seen_labels[sample_id]:
            # Have already seen label
            y = self.labels[sample_id]
            self.queried_oracle[t] = False
            return y
               
        # Otherwise need to query oracle
        y = self.labels[sample_id]
        self.seen_labels[sample_id] = True
        self.queried_oracle[t] = True
        return y

    def _calc_F_prob_est(self):
        """
        Calculates and estimate of the F-measure based on the probs
        and stores the result in self._F_prob_est
        """
        ##### CHANGE TO self.probs
        F_num = np.sum(self.p_probs * self.predictions)
        F_den = np.sum(self.p_probs * (1 - self.alpha) 
                       + self.alpha * self.predictions)
        self._F_prob_est = F_num/F_den
    
    def _calc_AOAIW_dist(self):
        """
        """
        
        if self.passive:
            self.pmf = self.density
            return
        
        # Easy vars
        epsilon = self.epsilon
        alpha = self.alpha
        thres = self.prob_threshold
        density = self.density
        p1 = self.p_probs
        p0 = 1 - self.p_probs
        t = self.t
        
        # Use most recent estimates of F and theta
        F_est = np.nan if t == 0 else self.F[t - 1]
                
        # Use an estimate for the F-measure based on the probs if it is np.nan
        if np.isnan(F_est) or F_est == 0:
            if self._F_prob_est is None:
                self._calc_F_prob_est()
            F_est = self._F_prob_est
        
        # FIXME clipping
        clip_eps = 1e-20
        # Predict positive pmf        
        pp_pmf = ( density * np.sqrt((alpha**2 * 
                                    F_est**2 * np.clip(p0, clip_eps, 1 - clip_eps) + (1 - F_est)**2 * np.clip(p1, clip_eps, 1 - clip_eps))) )
            
        # Predict negative pmf
        pn_pmf = ( density * (1 - alpha) * F_est * np.sqrt(np.clip(p1, clip_eps, 1 - clip_eps)) )
        
        self.pmf = (p1 >= thres) * pp_pmf + (p1 < thres) * pn_pmf
        # Normalize
        self.pmf = self.pmf/np.sum(self.pmf)
        # Weight by passive pmf
        self.pmf = epsilon * density + (1 - epsilon) * self.pmf          
        
    def reset(self):
        """
        Resets the instance to begin sampling again
        """
        self.t = 0
        
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0
        
        self.F = np.repeat(np.nan, self._max_iter)
        self.pmf = np.empty(self._num_pts)
        self.queried_oracle = np.repeat(False, self._max_iter)
        self.seen_labels = np.repeat(False, self._num_pts)
        
        self.density_1 = np.empty(self._num_pts)
        self.p_labels = (self.scores > self.score_threshold)*1
        self.p_probs = np.empty(self._num_pts)
        self._update_p_probs()
        
    def sample(self, n_iter):
        """
        Samples `n_iter` points
        """
        
        t_i = self.t
        t_f = n_iter + self.t
        
        assert t_f <= self.F.shape[0]
        
        for t in range(t_i, t_f):
            # Calculate pmf
            self._calc_AOAIW_dist()
            
            # Sample label and record weight
            sample_idx = self._sample_pt()
            w = self.density[sample_idx]/self.pmf[sample_idx]
            
            seen_before = self.seen_labels[sample_idx]
            
            # Query label and update seen status
            y = self._query_label(sample_idx)
            
            if self.debug == True:
                print("TP_term: {}, PP_term: {}, P_term: {}".format(self._TP_term, self._PP_term, self._P_term))
                print("Sampled label {} for point {}. Weight is {}.".format(y,sample_idx, w))
            
            if not seen_before:
                self.p_labels[sample_idx] = y
                if y == 1:
                    self._update_p_probs()
            self._update_F_terms(y, self.predictions[sample_idx], w)
            self._update_F()
                        
            self.t = self.t + 1

    def sample_until(self, n_goal):
        """
        Sample until `n_goal` labels are queried from the oracle
        """
        
        n_seen = np.sum(self.seen_labels)
        
        if n_seen >= n_goal:
            print("Have already queried {} labels from the oracle".format(n_seen))
            return
        
        if n_goal > self._max_iter:
            print("{} is greater than max_iter = {}".format(n_goal,self._max_iter))
            return
        
        while n_seen < n_goal:
            self.sample(1)
            n_seen = n_seen + self.queried_oracle[self.t - 1]*1
