import numpy as np
from .stratification import Strata
from scipy.special import expit

def verify_positive(value):
    """Throws exception if value is not positive"""
    if not value > 0:
        raise ValueError("expected positive integer")
    return value

def verify_predictions(predictions):
    """Ensures that predictions is stored as a numpy array and checks that
    all values are either 0 or 1.
    """
    # Check that it contains only zeros and ones
    predictions = np.array(predictions, copy=False)
    if not np.array_equal(predictions, predictions.astype(bool)):
        raise ValueError("predictions contains invalid values. " +
                         "The only permitted values are 0 or 1.")
    if predictions.ndim == 1:
        predictions = predictions[:,np.newaxis]
    return predictions

def verify_scores(scores):
    """Ensures that scores is stored as a numpy array and checks that all
    values are finite.
    """
    scores = np.array(scores, copy=False)
    if np.any(~np.isfinite(scores)):
        raise ValueError("scores contains invalid values. " +
                         "Please check that all values are finite.")
    if scores.ndim == 1:
        scores = scores[:,np.newaxis]
    return scores

def verify_consistency(predictions, scores, proba, opt_class):
    """Verifies that all arrays have consistent dimensions. Also verifies
    that the scores are consistent with proba.

    Returns
    -------
    proba, opt_class
    """
    if predictions.shape != scores.shape:
        raise ValueError("predictions and scores arrays have inconsistent " +
                         "dimensions.")

    n_class = scores.shape[1] if scores.ndim > 1 else 1

    # If proba not given, default to False for all classifiers
    if proba is None:
        proba = np.repeat(False, n_class)

    # If opt_class is not given, default to True for all classifiers
    if opt_class is None:
        opt_class = np.repeat(True, n_class)

    # Convert to numpy arrays if necessary
    proba = np.array(proba, dtype=bool, ndmin=1)
    opt_class = np.array(opt_class, dtype=bool, ndmin=1)

    if np.sum(opt_class) < 1:
        raise ValueError("opt_class should contain at least one True value.")

    if predictions.shape[1] != len(proba):
        raise ValueError("mismatch in shape of proba and predictions.")
    if predictions.shape[1] != len(opt_class):
        raise ValueError("mismatch in shape of opt_class and predictions.")

    for m in range(n_class):
        if (np.any(np.logical_or(scores[:,m] < 0, scores[:,m] > 1)) and proba[m]):
            warnings.warn("scores fall outside the [0,1] interval for " +
                          "classifier {}. Setting proba[m]=False.".format(m))
            proba[m] = False

    return proba, opt_class

def verify_unit_interval(value):
    """Throw an exception if the value is not on the unit interval [0,1].
    """
    if not (value >= 0 and value <= 1):
        raise ValueError("expected value on the interval [0, 1].")
    return value

def verify_boolean(value):
    """Throws an exception if value is not a bool
    """
    if type(value)!=bool:
        raise ValueError("expected boolean value.")
    return value

def verify_identifiers(identifiers, n_items):
    """Ensure that identifiers has a compatible length and that its elements
    are unique"""
    if identifiers is None:
        return identifiers

    identifiers = np.array(identifiers, copy=False)

    # Check length for consistency
    if len(identifiers) != n_items:
        raise ValueError("identifiers has inconsistent dimension.")

    # Check that identifiers are unique
    if len(np.unique(identifiers)) != n_items:
        raise ValueError("identifiers contains duplicate values.")

    return identifiers

def verify_strata(strata):
    """Ensure that input is of type `Strata`"""
    if strata is None:
        return strata
    if not isinstance(strata, Strata):
        raise ValueError("expected an instance of the Strata class")
    return strata

def scores_to_probs(scores, proba, eps=0.01):
    """Transforms scores to probabilities by applying the logistic function"""
    if np.any(~proba):
        # Need to convert some of the scores into probabilities
        probs = np.array(scores, copy=True)
        n_class = len(proba)
        for m in range(n_class):
            if not proba[m]:
                #TODO: incorporate threshold (currently assuming zero)
                # find most extreme absolute score
                max_extreme_score = max(np.abs(np.min(scores[:,m])),\
                                    np.abs(np.max(scores[:,m])))
                k = np.log((1-eps)/eps)/max_extreme_score # scale factor
                probs[:,m] = expit(k * scores[:,m])
        return probs
    else:
        return scores
