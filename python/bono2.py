# %%
# Import packages
from __future__ import absolute_import, division, print_function
import logging
import numpy as np

from demystifying import feature_extraction as fe, visualization
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# %%
# Define constants and cast empty array
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

# %%
# Randomly choose elements for features with correlation
# randomly choose elements
i = 1; idxrelat = [];
while i <= size(relatmat,1)
    if ismember(i,idxrelat)
        i = i+1;
        continue;
    end
    tmp = [i,find(abs(relatmat(i,:))>0.9)];
    idxrelat = union(idxrelat, randsample(tmp,length(tmp)-1));
    i = i+1;
end

idx = setdiff(idxorig,idxrelat); 
datamerge = [data(:,idx,1);data(:,idx,2)];

# idxorig = np.where(np.sum(data < cutoff, axis=(0, 2)))

# instead of this raw datamerge, I need to datamerge by relational matrix



samples = normalize(1/datamerge)
# samples = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=0)

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
feat_store = []

extractor = fe.RandomForestFeatureExtractor(samples=samples, labels=labels)
extractor.extract_features()

# %%

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

for feature_index, importance in postprocessor.get_important_features():
    if importance > 0.05:  # This cutoff limit is ad hoc and should be fine-tuned
        feat_store.append((feature_index, importance))
    # if importance < 0.5:  # This cutoff limit is ad hoc and should be fine-tuned
        # break
# logger.info("Feature %d has importance %.2f. Corresponds to residues %s", feature_index, importance,
#             feature_to_resids[int(feature_index)]
#             )

# TODO: merge 2 data sets into column on top of each other
# use labels with 1 and 0 to show which data set it belongs to
# get feature to resids from ( data[0][1] => protein-1_to_dna-2 ) relationship

# %%
# visualization.visualize([[postprocessor]], highlighted_residues=feat_store[0])
# visualization.visualize([[postprocessor]],)
fx = []  # List to store 'a' values
fy = []  # List to store 'b' values

for item in feat_store:
    a, b = item  # Unpack the tuple into variables 'a' and 'b'
    fx.append(a)  # Append 'a' to the 'x' list
    fy.append(b)  # Append 'b' to the 'y' list

# %%
plt.figure(figsize=(8, 3))
# plt.plot(range(35, 330), feat_store[0], linewidth=1.5)
plt.plot(fx, fy, linewidth=1.5)
plt.xlabel('Residue index')
plt.ylabel('Importance')
# plt.xlim([35, 329])
# plt.ylim([0, 1.1])
plt.show()
# %%
