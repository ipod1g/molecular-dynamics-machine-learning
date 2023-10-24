# %%
# Import packages
from __future__ import absolute_import, division, print_function
import random
import numpy as np
from demystifying import utils
from demystifying import feature_extraction as fe, visualization
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, normalize
from biopandas.pdb import PandasPdb
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

def get_feature_to_resids_from_pdb(n_features, pdb_file):
    pdb = PandasPdb()
    pdb.read_pdb(pdb_file)

    resid_numbers = np.unique(np.asarray(
        list(pdb.df['ATOM']['residue_number'])))


    # 520
    n_residues = len(resid_numbers)
    if n_features > (n_residues * (n_residues - 1)) // 2:
            raise ValueError("The number of features is incompatible with the number of residues")

    idx = 0
    feature_to_resids = np.empty((n_features, 2))
    print("resid_numbers size:", feature_to_resids.size)
    for res1 in range(n_residues):
        for res2 in range(res1, n_residues):
            feature_to_resids[idx, 0] = resid_numbers[res1]
            feature_to_resids[idx, 1] = resid_numbers[res2]
            idx += 1
    return feature_to_resids

#  %%
# get_feature_to_resids_from_pdb(n_features=520, pdb_file='structure-Swap.pdb')

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
# Randomly choose elements for features with correlation
# can loop for training set counts
i = 1
idxrelat = []
while i <= len(relatmat):
    if i in idxrelat:
        i += 1
        continue
    tmp = [i] + [j for j, val in enumerate(relatmat[i-1]) if abs(val) > 0.9]
    idxrelat = idxrelat + random.sample(tmp[1:], len(tmp)-1)
    i += 1

idx = np.setdiff1d(idxorig, idxrelat)
datamerge = np.concatenate((data[:, idx, 0], data[:, idx, 1]), axis=1)

# %%
samples = normalize(1/datamerge)  # X

# should be from feature_to_resids kind of label
labels = np.zeros((2 * N, 2))  # Y
labels[:N, 0] = 1
labels[N:, 1] = 1

# %%
feat_store = []

extractor = fe.RandomForestFeatureExtractor(
    samples=samples, labels=labels)
extractor.extract_features()

# %%
importance_primary = extractor.feature_importance
range_rows = importance_primary.shape[0] // 2  # Half of the range of rows

importance_populated = np.zeros(data.shape[1])
importance_populated[idx] = np.abs(importance_primary[:range_rows, 0])
# importance_populated[idx] = np.abs(importance_primary[:range_rows, 0])

# importance_reshaped = np.reshape(
#         importance_primary, (590, int(np.size(data,1) / 590)))
importance_reshaped = np.reshape(
    importance_populated, (590, int(np.size(data, 1) / 590)))

importance_normalized = normalize(importance_reshaped, norm='l2', axis=1)

importance_transposed = importance_normalized.T
scaler = MinMaxScaler()
scaled_importance = scaler.fit_transform(importance_transposed)


# %%
# Postprocess the results to convert importance per feature into importance per residue
postprocessor = extractor.postprocessing()
postprocessor.average()
postprocessor.persist()

# %%
# feat_store = []
for feature_index, importance in postprocessor.get_important_features():
    if importance < 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        # feat_store.append((feature_index, importance))
        logger.info("Feature %d has importance %.2f. Corresponds to residues xx", feature_index, importance,
                    # feature_to_resids[int(feature_index)]
                    )
    # if importance < 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        # break


# %%
# visualization.visualize([[postprocessor]], highlighted_residues=feat_store[0])


# %%
plt.figure(figsize=(8, 3))
plt.plot(range(35, 330), scaled_importance[0], linewidth=1.5)
plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([0, 1.1])
plt.show()
# %%
