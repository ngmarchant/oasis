import numpy as np

class Sawade:
    """
    Implements active importance sampling to estimate alpha-weighted F-measures.

    For details of the method, see the following paper:
    
        C. Sawade, N. Landwehr, and T. Scheffer, “Active Estimation of 
        F-Measures,” in Advances in Neural Information Processing Systems 23, 
        2010, pp. 2083–2091

    
    Input
    -----
    labels : int numpy array of length n
        array containing binary labels (assumed to be `0` or `1`) for each
        data point.

    probs : float numpy array of length n
        array containing positive class probabilities p(1|x) for each data 
        point.
    
    preds : int numpy array of length n
        array containing predicted binary labels for each data point.
    
    alpha : float, optional, default 0.5
        weight to use for the F-measure. It should be between 0 and 1, with a
        value of 1 corresponding to "precision", a value of 0 corresponding to 
        "recall" and a value of 0.5 corresponding to the balanced F-measure 
        (equal weight on precision and recall). Note that this parameterisation
        of the weighted F-measure is different to the usual one (see the paper 
        cited above for the definition).

    max_iter : int, optional, default None
        maximum number of iterations permitted (used to pre-allocate arrays for
        storing the sampling history). Defaults to the number of data points.
    
    epsilon : float, optional, default 1e-3
        epsilon-greedy parameter. Takes a value on the closed unit interval. 
        The "optimal" distribution is used with probability `1 - epsilon`, and 
        the passive distribution is used with probability `epsilon`. The 
        sampling is close to "optimal" for small epsilon.

    debug : bool, optional, default False
        if True, prints debugging information.
    """
    def __init__(self, labels, probs, preds, alpha = 0.5, 
                 epsilon = 1e-3, max_iter = None, debug = False):
        self.debug = debug
        self.labels = labels
        self.probs = probs
        self.alpha = alpha
        self.t = 0
        self.preds = preds
        self.epsilon = epsilon
        self._max_iter = max_iter
        self._num_pts = len(probs)
        self._F_prob_est = None
        
        if self._max_iter is None:
            self._max_iter = self._num_pts
            
        # Terms used to calculate F-measure (we update them iteratively)
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0
        
        # Array to record history of F-measure estimates
        self.F = np.repeat(np.nan, self._max_iter)
        # Array to record whether oracle was queried at each iteration
        self.queried_oracle = np.repeat(False, self._max_iter)
        # Array to record whether label for each point has been seen
        self.seen_labels = np.repeat(False, self._num_pts)
        
        self.pmf = np.empty(self._num_pts, dtype=float)
        self._initialise_pmf()
        
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
        return np.random.choice(self._num_pts, p = self.pmf)
    
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
            self.queried_oracle[t] = False
            return y
               
        # Otherwise need to query oracle
        y = self.labels[sample_idx]
        self.seen_labels[sample_idx] = True
        self.queried_oracle[t] = True
        return y
    
    def _calc_F_prob_est(self):
        """
        Calculates and estimate of the F-measure based on the probs
        and stores the result in self._F_prob_est
        """
        F_num = np.sum(self.probs * self.preds)
        F_den = np.sum(self.probs * (1 - self.alpha) 
                       + self.alpha * self.preds)
        self._F_prob_est = F_num/F_den
        
    def _initialise_pmf(self):
        """
        Calculates the asymptotically "optimal" instrumental distribution.
        """
        # Easy vars
        epsilon = self.epsilon
        alpha = self.alpha
        preds = self.preds
        p1 = self.probs
        p0 = 1 - p1
        num_pts = self._num_pts
        
        # Use an estimate for the F-measure based on the probs
        if self._F_prob_est is None:
            self._calc_F_prob_est()
        F_est = self._F_prob_est
        
        # Predict positive pmf
        pp_pmf = np.sqrt((alpha**2 * F_est**2 * p0 + (1 - F_est)**2 * p1))
        
        # Predict negative pmf
        pn_pmf = (1 - alpha) * F_est * np.sqrt(p1)
        
        # Calculate "optimal" pmf
        self.pmf = (preds == 1) * pp_pmf + (preds == 0) * pn_pmf

        # Normalise
        self.pmf = self.pmf/np.sum(self.pmf)
        
        # Weight by passive pmf
        self.pmf = ( epsilon * np.repeat(1/num_pts, num_pts) 
                        + (1 - epsilon) * self.pmf )
            
    def reset(self):
        """
        Resets the instance to begin sampling again
        """
        self.t = 0
           
        self._TP_term = 0
        self._PP_term = 0
        self._P_term = 0
        
        self.F = np.repeat(np.nan, self._max_iter)
        self.queried_oracle = np.repeat(False, self._max_iter)
        self.seen_labels = np.repeat(False, self._num_pts)        
        
    def sample(self, n_iter):
        """
        Samples `n_iter` points
        """
        t_i = self.t
        t_f = n_iter + self.t
        
        assert t_f <= self.F.shape[0]
        
        for t in range(t_i, t_f):
           
            # Sample label and record weight
            sample_idx = self._sample_pt()
            w = (1/self._num_pts)/self.pmf[sample_idx]
            
            # Query label
            y = self._query_label(t, sample_idx)
            
            # Get score
            s = self.probs[sample_idx]
            
            if self.debug == True:
                print("Sampled label {} for point {}. Weight is {}".format(y,sample_idx,w))
            
            self._update_F_terms(y, self.preds[sample_idx], w)
            self.F[t] = self._calc_F()
            
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
