import numpy as np

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

def verify_consistency(predictions, scores, proba):
    """Verifies that all arrays have consistent dimensions. Also verifies
    that the scores are consistent with proba. Returns proba.
    """
    if predictions.shape != scores.shape:
        raise ValueError("predictions and scores arrays have inconsistent " +
                         "dimensions.")

    n_class = scores.shape[1] if scores.ndim > 1 else 1

    # If proba not given, default to False for all classifiers
    if proba is None:
        return np.repeat(False, n_class)

    # Ensure proba is a numpy array (important for case of one classifier)
    proba = np.array(proba, dtype=bool, ndmin=1)

    if predictions.shape[1] != len(proba):
        raise ValueError("length of proba and number of columns in " +
                         "should match.")

    for m in range(n_class):
        if (np.any(np.logical_or(scores[:,m] < 0, scores[:,m] > 1)) and proba[m]):
            warnings.warn("scores fall outside the [0,1] interval for " +
                          "classifier {}. Setting proba[m]=False.".format(m))
            proba[m] = False
    return proba

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
    if isinstance(x, oasis.Strata):
        raise ValueError("expected an instance of the Strata class")
    return strata
