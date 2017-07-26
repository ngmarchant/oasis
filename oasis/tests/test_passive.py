import numpy as np
import pytest
from oasis import PassiveSampler

# test invalid alpha
# test predictions with invalid shape
# test oracle returns invalid label
# test invalid max_iter
# test sampling with/without replacement

def test_predictions_contains_invalid_value():
    predictions = np.array([0, 1, 0, 1, -1000000])

    with pytest.raises(ValueError):
        smplr = PassiveSampler(alpha=0.5, \
                               predictions=predictions, \
                               oracle=lambda x: 1)

def test_using_identifiers_argument():
    predictions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels = {'b': 0, 'a': 0, 'f': 1, 'g': 1, 'h': 1, 'c': 1, 'e': 0, 'd': 0}
    def oracle(idx):
        return labels.get(idx)
    smplr = PassiveSampler(alpha=0.5, \
                           predictions=predictions, \
                           oracle=oracle, \
                           identifiers=list(labels.keys()), \
                           replace=False)

class TestPassiveSampler:
    def setup_class(self):
        self.alpha = 0.5
        self.n_items = 8
        self.predictions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.labels = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        self.true_F1 = 0.75
        def oracle(identifier):
            return self.labels[identifier]
        self.smplr = PassiveSampler(alpha=self.alpha, \
                                    predictions=self.predictions, \
                                    oracle=oracle,
                                    replace=False)

    def test_reset(self):
        # reset
        self.smplr.reset()

        # check that arrays/variables are indeed reset
        assert self.smplr.t_ == 0
        np.testing.assert_equal(self.smplr.cached_labels_, \
                                np.repeat(np.nan, self.n_items))
        np.testing.assert_equal(self.smplr._queried_oracle, \
                                np.repeat(False, self.n_items))
        np.testing.assert_equal(self.smplr._estimate.ravel(), \
                                np.repeat(np.nan, self.n_items))

    def test_convergence(self):
        self.smplr.reset()
        # sample all items
        self.smplr.sample(self.n_items)

        # estimate should equal true value
        assert self.smplr.estimate_[-1] == self.true_F1

        # cached labels should equal true ones
        np.testing.assert_equal(self.smplr.cached_labels_,\
                                self.labels)

        # should have done n_items iterations
        assert self.smplr.t_ == self.n_items

    def test_sample_past_maximum_iterations(self):
        self.smplr.reset()

        self.smplr.sample(self.n_items)

        with pytest.raises(Exception):
            self.smplr.sample(1)

    def test_sample_with_replacement(self):
        self.smplr.reset()
        self.smplr.replace = True
        self.smplr.sample(self.n_items)
        # Can't check any condition
