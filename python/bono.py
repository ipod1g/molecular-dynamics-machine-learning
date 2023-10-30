# %%
# Import packages
from __future__ import absolute_import, division, print_function
import random
import numpy as np
from demystifying import feature_extraction as fe
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import logging

# %%
# Define constants and cast empty array
logger = logging.getLogger("demo")
logger.setLevel('INFO')

group = ['GC', 'Swap']  # equivalent of labels
groupn = ["Natural G/C", "-1/+1-swap"]  # additional description for above

# Found from the size of mindisres data
N = 1204
M = 10620

data = np.zeros((N, M, len(group)))

# %%
# Read data files
for k in range(2):
    file_path = 'mindisres-' + group[k] + '.xvg'
    data[:, :, k] = np.loadtxt(
        file_path, skiprows=24, usecols=range(1, M+1))

# %%
# Setup for preprocess
data = data * 10
datamerge = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=0)
dataNorm = normalize(1.0 / datamerge)
R = np.corrcoef(dataNorm.T)
relatmat = np.triu(R, k=1)
cutoff = 5  # remove residue candidates with heavy-atom min distance > 5Å
idxorig = np.where(np.sum(data < cutoff, axis=(0, 2)))

# %%
# Shuffling
# can loop for training set counts
training_iteration = 1
iptglobal = []

for t in range(training_iteration):
    i = 1
    idxrelat = []
    while i <= len(relatmat):
        if i in idxrelat:
            i += 1
            continue
        tmp = [i] + \
            [j for j, val in enumerate(relatmat[i-1]) if abs(val) > 0.9]
        idxrelat = idxrelat + random.sample(tmp, len(tmp)-1)
        i += 1

    idx = np.setdiff1d(idxorig, idxrelat)
    datamerge2 = np.concatenate((data[:, idx, 0], data[:, idx, 1]), axis=0)

    samples = normalize(1.0/datamerge2)  # X

    labels = np.zeros((2 * N, 2))  # Y
    labels[:N, 0] = 1
    labels[N:, 1] = 1

    extractor = fe.RandomForestFeatureExtractor(
        samples=samples, labels=labels,
        # classifier_kwargs={'n_estimators': 20},
    )
    extractor.extract_features()

    importance_primary = extractor.feature_importance
    range_rows = importance_primary.shape[0]

    importance_populated = np.zeros(data.shape[1])  # 10620
    # not sure why I have 2? must be  from labels
    average_importance = (
        importance_primary[:range_rows, 0] + importance_primary[:range_rows, 1]) / 2
    importance_populated[idx] = np.abs(average_importance)
    importance_reshaped = np.reshape(
        importance_populated, (590, int(np.size(data, 1) / 590)))
    importance_normalized = normalize(importance_reshaped, norm='l2', axis=1)
    importance_transposed = importance_normalized.T
    iptorig = np.sum(importance_transposed, axis=0)
    iptorig = iptorig.reshape(1, -1)
    ipt = normalize(iptorig, norm='max')
    iptglobal.extend([ipt])

# %%
# Get mean importance from all training sets
iptglobal = np.array(iptglobal)
mean_trained_importance = np.mean(iptglobal, axis=0)

# %%
plt.figure(figsize=(8, 3))
plt.plot(range(35, 330), mean_trained_importance[0, 0:295], linewidth=1.5)
plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([0, 1.1])
plt.savefig('0-295.png')
plt.show()
# %%
plt.figure(figsize=(8, 3))
plt.plot(range(35, 330), mean_trained_importance[0, 295:590], linewidth=1.5)
plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([0, 1.1])
plt.savefig('295-590.png')
plt.show()

# %%
