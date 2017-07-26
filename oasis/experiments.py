import numpy as np
import tables
import time
import logging
import warnings

def repeat_expt(sampling_obj, num_expts, num_labels, output_file = None):
    """
    Input
    -----
    sampling_obj

    num_expts        number of expts to run

    num_labels       number of labels to query from the oracle in each expt


    sampling_obj must have:
    sample_until method
    reset method
    F property
    """

    FILTERS = tables.Filters(complib='zlib', complevel=5)

    max_iter = sampling_obj._max_iter
    if max_iter < num_labels:
        raise ValueError("Cannot query {} labels. {} supports only {} \
            iterations".format(num_labels, sampling_obj.__name__, max_iter))

    if output_file is None:
        # Use current date/time as filename
        output_file = 'expt_' + time.strftime("%d-%m-%Y_%H:%M:%S") + '.h5'
    logging.info("Writing output to {}".format(output_file))

    f = tables.open_file(output_file, mode='w', filters=FILTERS)
    float_atom = tables.Float64Atom()
    bool_atom = tables.BoolAtom()
    int_atom = tables.Int64Atom()

    array_F = f.create_earray(f.root, 'F_measure', float_atom, (0, num_labels))
    array_s = f.create_carray(f.root, 'n_iterations', int_atom, (num_expts, 1))
    array_t = f.create_carray(f.root, 'CPU_time', float_atom, (num_expts, 1))

    logging.info("Starting {} experiments".format(num_expts))
    for idx in range(num_expts):
        if idx%100 == 0:
            logging.info("Completed {} of {} experiments".format(idx, num_expts))
        ti = time.process_time()
        sampling_obj.reset()
        sampling_obj.sample_distinct(num_labels)
        tf = time.process_time()
        if hasattr(sampling_obj, 'queried_oracle_'):
            array_F.append([sampling_obj.estimate_[sampling_obj.queried_oracle_]])
        else:
            array_F.append([sampling_obj.estimate_])
        array_s[idx] = sampling_obj.t_
        array_t[idx] = tf - ti
    f.close()

    logging.info("Completed all experiments")

def process_expt(h5_path, verbose = False, inmemory = True, ignorenan = False):
    """
    Assumes h5 file has table called `F_measure`

    Input
    -----
    h5_path : string

    verbose : boolean

    inmemory : boolean

    Output
    ------
    F_mean : float numpy.ndarray

    F_var : float numpy.ndarray
    """

    logging.info("Reading file at {}".format(h5_path))
    h5_file = tables.open_file(h5_path, mode = 'r')

    F = h5_file.root.F_measure

    n_labels = F.shape[1]
    n_expt = F.shape[0]
    mean_n_iterations = np.sum(h5_file.root.n_iterations)/n_expt

    if hasattr(h5_file.root, 'CPU_time'):
        CPU_time = h5_file.root.CPU_time
        mean_CPU_time = np.mean(CPU_time)
        var_CPU_time = np.var(CPU_time)
    else:
        mean_CPU_time = None
        var_CPU_time = None
        mean_CPU_time_per_iteration = None

    F_mean = np.empty(n_labels, dtype='float')
    F_var = np.empty(n_labels, dtype='float')
    F_stderr = np.empty(n_labels, dtype='float')
    n_sample = np.empty(n_labels, dtype='int')
    if inmemory:
        F_mem = F[:,:]

    logging.info("Beginning processing".format())
    for t in range(n_labels):

        if t%1000 == 0:
            logging.info("Processed {} of {} experiments".format(t, n_labels))
        if inmemory:
            temp = F_mem[:,t]
        else:
            temp = F[:,t]
        if ignorenan:
            n_sample[t] = np.sum(~np.isnan(temp))
            # Expect to see RuntimeWarnings if array contains all NaNs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                F_mean[t] = np.nanmean(temp)
                F_var[t] = np.nanvar(temp)
                F_stderr[t] = np.sqrt(F_var[t]/n_sample[t])
        else:
            n_sample[t] = len(temp)
            F_mean[t] = np.mean(temp)
            F_var[t] = np.var(temp)
            F_stderr[t] = np.sqrt(F_var[t]/n_sample[t])

    logging.info("Processing complete".format())

    h5_file.close()

    return {'mean': F_mean,
            'variance': F_var,
            'std_error': F_stderr,
            'n_samples': n_sample,
            'n_expts': n_expt,
            'n_labels': n_labels,
            'mean_CPU_time': mean_CPU_time,
            'var_CPU_time': var_CPU_time,
            'mean_n_iterations': mean_n_iterations,
            'h5_path': h5_path}

class DataError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class Data:
    """

    """
    def __init__(self):
        self.features = None
        self.scores = None
        self.labels = None
        self.probs = None
        self.preds = None
        self.threshold = None
        self.TP = None
        self.FP = None
        self.TN = None
        self.FN = None
        self.F1_measure = None
        self.precision = None
        self.recall = None

    def read_h5(self, h5_path, load_features=False):

        h5_file = tables.open_file(h5_path, mode = 'r')

        if load_features and hasattr(h5_file.root, "features"):
            self.features = h5_file.root.features[:,:]
            self.num_fts = h5_file.root.features.shape[1]
        if hasattr(h5_file.root, "labels"):
            self.labels = h5_file.root.labels[:]
        if hasattr(h5_file.root, "scores"):
            self.scores = h5_file.root.scores[:]
        if hasattr(h5_file.root, "probs"):
            self.probs = h5_file.root.probs[:]
        if hasattr(h5_file.root, "preds"):
            self.preds = h5_file.root.preds[:]
            self.num_items = len(self.preds)

        h5_file.close()

    def scores_to_preds(self, threshold = 0, use_probs = True):
        """
        use_probs : boolean, default True
            if True, use probabilities for predictions, else use scores.
        """
        self.threshold = threshold

        if use_probs:
            if self.probs is None:
                raise DataError("Probabilities are not available to make "
                                "predictions.")
            else:
                word = "probabilities"
                scores = self.probs
        else:
            if self.scores is None:
                raise DataError("Scores are not available to make predictions.")
            else:
                word = "scores"
                scores = self.scores

        if threshold > np.max(scores) or threshold < np.min(scores):
            warnings.warn("Threshold {} is outside the range of the "
                          "{}.".format(self.threshold, word))

        if self.preds is not None:
            warnings.warn("Overwriting predictions")
        self.preds = (scores >= threshold)*1

    def calc_confusion_matrix(self, printout = False):
        """
        Calculates number of TP, FP, TN, FN
        """
        if self.labels is None:
            raise DataError("Cannot calculate confusion matrix before data "
                            "has been read.")

        if self.preds is None:
            raise DataError("Predictions not available. Please run "
                            "`make_predictions` before calculating confusion "
                            "matrix")

        self.TP = np.sum(np.logical_and(self.preds == 1, self.labels == 1))
        self.TN = np.sum(np.logical_and(self.preds == 0, self.labels == 0))
        self.FP = np.sum(np.logical_and(self.preds == 1, self.labels == 0))
        self.FN = np.sum(np.logical_and(self.preds == 0, self.labels == 1))

        if printout:
            print("Contingency matrix is:")
            print("----------------------")
            print("TP: {} \t FN: {}".format(self.TP,self.FN))
            print("FP: {} \t TN: {}".format(self.FP,self.TN))
            print("\n")


    def calc_true_performance(self, printout = False):
        """
        Evaluate precision, recall and balanced F-measure
        """

        try:
            self.calc_confusion_matrix(printout = False)
        except DataError as e:
            print(e.msg)
            raise

        if self.TP + self.FP == 0:
            self.precision = np.nan
        else:
            self.precision = self.TP / (self.TP + self.FP)

        if self.TP + self.FN == 0:
            self.recall = np.nan
        else:
            self.recall = self.TP / (self.TP + self.FN)

        if self.precision + self.recall == 0:
            self.F1_measure = np.nan
        else:
            self.F1_measure = ( 2 * self.precision * self.recall /
                                 (self.precision + self.recall) )

        if printout:
            print("True performance is:")
            print("--------------------")
            print("Precision: {} \t Recall: {} \t F1 measure: {}".format(self.precision, self.recall, self.F1_measure))
