import numpy as np
from oasis.experiments import Data
from scipy.special import expit
from oasis.oasis import OASISSampler
#from oasis.sawade import ImportanceSampler
from sawade import ImportanceSampler
data = Data()


data.read_h5('Amazon-GoogleProducts-test.h5')


def oracle(idx):
    return data.labels[idx]


alpha = 0.5


def scores2probs(scores, eps=0.01):
    max_extreme_score = np.max(np.abs(scores))
    k = np.log((1 - eps) / eps) / max_extreme_score  # scale factor
    return expit(k * scores)


probs = scores2probs(data.scores)

print(data.preds.shape, data.preds.sum())
print(data.scores.shape, data.scores.min(), data.scores.max(), data.scores.mean())
smplr =ImportanceSampler(alpha, data.preds, probs, oracle)
smplr.sample_distinct(5000)
#print(smplr.FN, smplr.FP, smplr.TP)
print(smplr.estimate())