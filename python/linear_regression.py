# %%
# Import packages
from __future__ import absolute_import, division, print_function
import random
import numpy as np
from demystifying import feature_extraction as fe
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import logging
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

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
        idxrelat = idxrelat + random.sample(tmp, len(tmp) - 1)
        i += 1

    idx = np.setdiff1d(idxorig, idxrelat)
    datamerge2 = np.concatenate((data[:, idx, 0], data[:, idx, 1]), axis=0)

    X = normalize(1.0/datamerge2)  # X

    # labels = np.zeros((2 * N, 2))  # Y
    # labels[:N, 0] = 1
    # labels[N:, 1] = 1
    Y = np.concatenate([
        np.ones((data.shape[0], 1)),
        np.zeros((data.shape[0], 1))
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42)

    # Define a list of lambda values for regularization
    lambda_values = np.logspace(-6, -0.5, len(idx))

    # Initialize a logistic regression model with L1 (Lasso) regularization
    model = LogisticRegressionCV(
        Cs=1 / lambda_values,
        cv=len(lambda_values),
        penalty='l1',
        solver='saga',
        max_iter=5000,
        # max_iter=10000,
        tol=1e-8,
        scoring='neg_log_loss',  # To match MATLAB's logarithmic loss
        random_state=42  # For reproducibility
    )

    # Fit the logistic regression model with different lambda values

    # Convert one-hot encoded Y_train to 1D if necessary
    if Y_train.ndim > 1 and Y_train.shape[1] > 1:
        Y_train = np.argmax(Y_train, axis=1)

    # Train the model on the training data
    model.fit(X_train, Y_train)

    labels = model.predict(X_test)

    importance_populated = np.zeros(data.shape[1])  # 10620
    importance_populated[idx] = np.abs(model.coef_)

    importance_reshaped = np.reshape(
        importance_populated, (590, int(np.size(data, 1) / 590)))

    importance_normalized = normalize(
        importance_reshaped, norm='l2', axis=1)
    importance_transposed = importance_normalized.T
    iptorig = np.nansum(importance_transposed, axis=0)
    iptorig = iptorig.reshape(1, -1)
    ipt = normalize(iptorig, norm='max')
    # Append ipt to iptglob
    iptglobal.append(ipt)


# %%
# Get mean importance from all training sets
iptglobal = np.array(iptglobal)
mean_trained_importance = np.mean(iptglobal, axis=0)

# %%
plt.figure(figsize=(8, 3))

# Plot the first set of data (0:295) in blue with a solid line
plt.plot(range(35, 330),
         mean_trained_importance[0, 0:295], linewidth=1.5, label='0-295 idx', color='#429EBD')

# Plot the second set of data (295:590) in red with a dashed line
plt.plot(range(35, 330), mean_trained_importance[0, 295:590],
         linewidth=1.5, label='295-590 idx', linestyle='--', color='#FFB347')

plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([0, 1.1])

plt.legend()
plt.savefig('combined_GC-Swap.png')
plt.show()


# %%
