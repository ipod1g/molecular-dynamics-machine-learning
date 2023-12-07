# %%
# Import packages
from __future__ import absolute_import, division, print_function
import logging
import random
import numpy as np
from demystifying import feature_extraction as fe
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from demystifying import relevance_propagation as relprop

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
    file_path = './mindisres-' + group[k] + '.xvg'
    data[:, :, k] = np.loadtxt(
        file_path, skiprows=24, usecols=range(1, M+1))

# %%
# Setup for preprocess
data = data * 10
datamerge = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=0)
dataNorm = normalize(1.0 / datamerge)
R = np.corrcoef(dataNorm.T)
relatmat = np.triu(R, k=1)
lower_distance_cutoff = 50  # remove residue candidates with heavy-atom min distance > 5Å
correlation_cutoff = 0.9
idxorig = np.where(np.sum(data < lower_distance_cutoff, axis=(0, 2)))

# %%
# Shuffling
# Can shuffle once or for each iteration
i = 1
idxrelat = []
while i <= len(relatmat):
    if i in idxrelat:
        i += 1
        continue
    tmp = [i] + \
        [j for j, val in enumerate(relatmat[i-1]) if abs(val) > correlation_cutoff] 
    idxrelat = idxrelat + random.sample(tmp, len(tmp)-1)
    i += 1

idx = np.setdiff1d(idxorig, idxrelat)
datamerge2 = np.concatenate((data[:, idx, 0], data[:, idx, 1]), axis=0)

samples = normalize(1.0/datamerge2)  # X

labels = np.zeros((2 * N, 2))  # Y
labels[:N, 0] = 1
labels[N:, 1] = 1

# %%
# Setup models for training
number_of_k_splits = 10
number_of_iterations = 1

kwargs = {'samples': samples, 'labels': labels,
          'use_inverse_distances': True,
          'n_splits': number_of_k_splits, 'n_iterations': number_of_iterations, 'scaling': True}

feature_extractors = [
    fe.RandomForestFeatureExtractor(one_vs_rest=True, classifier_kwargs={
        'n_estimators': 10}, **kwargs),
    # 'n_estimators': 500}, **kwargs),
    fe.PCAFeatureExtractor(variance_cutoff=0.75, **kwargs),
    fe.RbmFeatureExtractor(relevance_method="from_components", **kwargs),
    # fe.MlpAeFeatureExtractor(activation=relprop.relu, classifier_kwargs={
    #     'solver': 'adam',
    #     'hidden_layer_sizes': (100,)
    # }, **kwargs),
    # fe.KLFeatureExtractor(**kwargs),
    # fe.MlpFeatureExtractor(classifier_kwargs={'hidden_layer_sizes': (120,),
    #                                           'solver': 'adam',
    #                                           'max_iter': 1000000
    #                                           }, activation=relprop.relu, **kwargs),
]

# %%
iptglobal = []
for extractor in feature_extractors:
    extracted = []
    for t in range(number_of_iterations):
        extractor.extract_features()

        importance_primary = extractor.feature_importance
        rows, cols = importance_primary.shape
        importance_populated = np.zeros(data.shape[1])  # 10620

        # average_importance = np.zeros(importance_primary.shape[0])
        # for col in range(cols):
        #     average_importance += importance_primary[:rows, col]
        # average_importance /= cols

        importance_populated[idx] = np.abs(importance_primary[:rows, 0])
        # importance_populated[idx] = np.abs(average_importance)
        importance_reshaped = np.reshape(
            importance_populated, (590, int(np.size(data, 1) / 590)))
        importance_normalized = normalize(
            importance_reshaped, norm='l2', axis=1)
        importance_transposed = importance_normalized.T
        iptorig = np.sum(importance_transposed, axis=0)
        iptorig = iptorig.reshape(1, -1)
        ipt = normalize(iptorig, norm='max')
        extracted.append(ipt)
    iptglobal.append(extracted)

# %%
# Get mean importance from all training sets
iptglobal = np.array(iptglobal)


def get_mean_trained_importance(modelName):
    model_name_to_index = {
        'rf': 0,
        'pca': 1,
        'rbm': 2,
        'mlp-ae': 3,
        'kl': 4,
        'mlp': 5,
    }
    if modelName in model_name_to_index:
        model_index = model_name_to_index[modelName]
        mean_trained_importance = np.mean(
            iptglobal[model_index, :, :, :], axis=0)
        return mean_trained_importance
    else:
        # Handle the case when modelName is not found
        raise ValueError(f"Model name '{modelName}' not recognized")


# %%
modelDisplay = "rf"
mean_trained_importance = get_mean_trained_importance(modelDisplay)
plt.figure(figsize=(8, 3))

# Plot the first set of data (0:295) in blue with a solid line
plt.plot(range(35, 330),
         mean_trained_importance[0, 0:295], linewidth=1.5, label='0-295 idx', color='#429EBD')

# Plot the second set of data (295:590) in red with a dashed line
plt.plot(range(35, 330), mean_trained_importance[0, 295:590],
         linewidth=1.5, label='295-590 idx', linestyle='--', color='#FFB347')
# plt.title(modelDisplay)
plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([0, 1.1])

plt.legend()
plt.savefig('combined_GC-Swap.png')
plt.show()


# %%
