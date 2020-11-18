from scipy.stats import expon
import matplotlib.pyplot as plt
from refactored.stratification import stratify_by_cum_sqrt_f_method, stratify_by_equal_size_method, Strata
from collections import Counter

rv = expon()
obs = rv.rvs(10000)


fig, axes = plt.subplots(1, 2)
axes = axes.flatten()
allocation0 = stratify_by_equal_size_method(obs)
allocation1 = stratify_by_cum_sqrt_f_method(obs)
strats0 = Strata(allocation1)

print(Counter(allocation1))
axes[0].hist(allocation0)
axes[1].hist(allocation1)
plt.show()
