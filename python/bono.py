# %%
from __future__ import absolute_import, division, print_function
import logging
import numpy as np

from demystifying import feature_extraction as fe, visualization


# %%
logger = logging.getLogger("bono")
logger.setLevel('INFO')

group = ['GC', 'Swap']  # equivalent of labels
groupn = ["Natural G/C", "-1/+1-swap"]  # additional description for above
columnN = 295*2*11*2  # From the readme = 12980

# data = np.zeros((1204, columnN, len(group)))

N = 1204
# M = 10620
M = 2000  # reduce columns to lessen strain when testing
data = np.zeros((N, M, 2))

# To read xvg of both files
for k in range(2):
    file_path = 'mindisres-' + group[k] + '.xvg'
    data[:, :, k] = np.loadtxt(
        file_path, skiprows=24, usecols=range(1, M+1))

data = data * 10

cutoff = 5  # remove residue candidates with heavy-atom min distance > 5Å

# %%

# idxorig = np.where(np.sum(data < cutoff, axis=(0, 2)))
samples = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=0)
# %%

# get indices where data is less than cutoff - to filter out
idxorig = np.where(np.sum(data < cutoff, axis=(0, 2)) > 0)[0]
# TODO: need to apply this to both samples and labels

labels = np.zeros((2 * N, 2))
labels[:N, 0] = 1
labels[N:, 1] = 1

# %%

# our data is already clustered => 10000 frames per cluster as 'state'
# 3000000 frames in total => 300 clusters

# feature_to_resids = dg.feature_to_resids
# For distance based features this will be a (n_features, 2) matrix, where you list the two residues that you measure the distance between.

# data[0][0] => protein-1_to_dna-1
# data[0][1] => protein-2_to_dna-1
# ...

extractor = fe.RandomForestFeatureExtractor(samples=samples, labels=labels)
extractor.extract_features()

# Postprocess the results to convert importance per feature into importance per residue
postprocessor = extractor.postprocessing()
postprocessor.average()
postprocessor.persist()

# %%
feat_store = []
# need to convert from ( data[0][1] => protein-1_to_dna-2 ) relationship
# must pay attention to separation between 2 data set
# must also note that in samples the row doesnt tell the resids
feat_resids = []

for feature_index, importance in postprocessor.get_important_features(sort=True):
    if importance > 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        feat_store.append((feature_index, importance))
    # if importance < 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        # break
# logger.info("Feature %d has importance %.2f. Corresponds to residues %s", feature_index, importance,
#             feature_to_resids[int(feature_index)]
#             )

logger.info("Done")

# TODO: merge 2 data sets into column on top of each other
# use labels with 1 and 0 to show which data set it belongs to
# get feature to resids from ( data[0][1] => protein-1_to_dna-2 ) relationship

# %%
# visualization.visualize([[postprocessor]], highlighted_residues=feat_store[0])
visualization.visualize([[postprocessor]])

# %%
