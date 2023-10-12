# %%
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# %%
group = ['GC', 'Swap']
groupn = ["Natural G/C", "-1/+1-swap"]

N = 1204
M = 10620
# M = 5000  # reduce columns to lessen strain when testing
data = np.zeros((N, M, 2))

for k in range(2):
    file_path = 'mindisres-' + group[k] + '.xvg'
    data[:, :, k] = np.loadtxt(
        file_path, skiprows=24, usecols=range(1, M+1))


data = data * 10

# Preprocessing
# %%
cutoff = 5
idxorig = np.where(np.sum(data < cutoff, axis=(0, 2)))

datamerge = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=0)

dataNorm = normalize(1.0 / datamerge)
R = np.corrcoef(dataNorm.T)
relatmat = np.triu(R, k=1)

# %%
# Logistic regression
iptglob = []

trainRepeat = 2

# Train the model 10 times with randomly chosen features for statistics
for t in range(trainRepeat):
    idxrelat = set()
    i = 1
    while i < relatmat.shape[0]:
        if i in idxrelat:
            i += 1
            continue
        tmp = [i] + list(np.where(np.abs(relatmat[i, :]) > 0.9)[0])
        idxrelat.update(np.random.choice(
            np.array(tmp).flatten(), size=len(tmp) - 1, replace=False))
        # idxrelat.update(np.random.choice(
        #     tmp, size=len(tmp) - 1, replace=False))
        i += 1

    idx = np.setdiff1d(idxorig, list(idxrelat))
    datamerge = np.concatenate([data[:, idx, 0], data[:, idx, 1]])

    X = normalize(1. / datamerge)
    Y = np.concatenate(
        (np.ones((data.shape[0], 1)), np.zeros((data.shape[0], 1))))

    test_size = 0.4
    Y_size = Y.shape[0]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)

    x_train = x_train.astype(int)
    x_test = x_test.astype(int)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    X = X.T
    lambda_vals = np.logspace(-6, -0.5, len(idx))

    Mdl = LogisticRegression(penalty='l1', solver='saga',
                             C=1.0 / lambda_vals[0], max_iter=1000, tol=1e-8)

    Mdl.fit(x_train, y_train)
    # needs to be datamerge columns but its 1 now,,,
    labels = Mdl.predict(x_test)
    L = Mdl.score(x_test, y_test)

    # Mdl.fit(X[idxTrain, :], Y[idxTrain])
    # labels = Mdl.predict(X[idxTest, :])
    # L = Mdl.score(X[:, idxTest], Y[idxTest])

    importance = np.zeros((1, data.shape[1]))
    # importance[idx] = np.abs(Mdl.coef_[:,1])
    importance[0, idx] = np.abs(Mdl.coef_[0])
    reshaped_importance = np.reshape(
        importance, (590, int(data.shape[1] / 590)))
    normalized_importance = normalize(reshaped_importance, norm='l2', axis=0)
    iptorig = np.nansum(normalized_importance, axis=1)
    ipt = normalize(iptorig)

    iptglob[t, :] = ipt

    # reference
    # importance = np.zeros(data.shape[1])
    # importance[idx] = np.abs(Mdl.coef_[:,1])
    # importance = np.reshape(590, (-1, 1))
    # importance = normalize(importance, norm='l2', axis=0)
    # iptorig = np.nansum(importance, axis=1)
    # ipt = normalize(iptorig.reshape(-1, 1), axis=0, norm='max')
    # iptglob[t, :] = ipt

# %%
ipt = np.mean(iptglob, axis=0)

plt.figure(figsize=(8, 3))
plt.plot(range(35, 330), ipt[:295], linewidth=1.5)
plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([-0.1, 1.1])
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(range(35, 330), ipt[295:590], linewidth=1.5)
plt.xlabel('Residue index')
plt.ylabel('Importance')
plt.xlim([35, 329])
plt.ylim([-0.1, 1.1])
plt.show()

# %%
