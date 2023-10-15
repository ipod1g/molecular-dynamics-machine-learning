from __future__ import absolute_import, division, print_function

import logging

from demystifying import feature_extraction as fe, visualization
from demystifying.data_generation import DataGenerator

import numpy as np

logger = logging.getLogger("bono")
logger.setLevel('INFO')

group = ['GC', 'Swap']  # equivalent of labels
groupn = ["Natural G/C", "-1/+1-swap"]  # additional description for above
# columnN = 295*2*11*2  # From the readme = 12980
columnN = 295*2*11*2  # From the readme = 12980

data = np.zeros((1204, columnN, len(group)))

# our data is already clustered => 10000 frames per cluster as 'state'
# 3000000 frames in total => 300 clusters

# feature_to_resids = dg.feature_to_resids
# For distance based features this will be a (n_features, 2) matrix, where you list the two residues that you measure the distance between.

# data[0][1] => protein-1_to_dna-2 

# To read xvg of both files
for k in range(len(group)):
    file_path = 'mindisres-' + group[k] + '.xvg'
    data[:, :, k] = np.loadtxt(
        file_path, skiprows=24, usecols=range(1, columnN + 1))
    # data.append(np.loadtxt(file_path, skiprows=24, usecols=range(1, columnN)))

data = np.array(data) * 10
logger.info(data)

file_output_path = 'output/gromacs-data.xvg'

# Save the data to the .xvg file
# np.savetxt(file_output_path, data, delimiter='\t',
#            fmt='%.6f', header=data.shape)
np.savetxt(file_output_path, data.reshape(-1, 1), delimiter='\t',
           fmt='%.6f')


# Preprocessing
cutoff = 5  # remove residue candidates with heavy-atom min distance > 5Å

# get indices where data is less than cutoff - to filter out
idxorig = np.where(np.sum(data < cutoff, axis=(0, 2)) > 0)[0]

# Concatenate the two slices of data along the first axis
datamerge = np.concatenate((data[:, :, 0], data[:, :, 1]), axis=0)


logger.info("Done")



# TODO: merge 2 data sets into column on top of each other
# use labels with 1 and 0 to show which data set it belongs to
# get feature to resids from ( data[0][1] => protein-1_to_dna-2 ) relationship