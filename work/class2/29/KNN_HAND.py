# libraries

import numpy as np
import os
import pandas as pd
import re
from io import StringIO
from sklearn.model_selection import StratifiedKFold
#os.getcwd()
#os.chdir("ur folder directory contains training/testDigits")

# Get labels of y from file names

def getLabel(folder):
    names = os.listdir(folder)
    n_names = len(names)
    label = np.zeros(n_names)
    for i in range(n_names):
        label[i] = int(re.split('_|\.', names[i])[0])
    return label

# reshape data matrix to n X (32 X 32)

def b2vM(folder):
    for i, files in enumerate(os.listdir(os.path.join(os.getcwd(), folder))):
        with open(os.path.join(folder, files)) as f:
            temp = f.read()
            n = len(temp.splitlines())
            data = pd.read_fwf(StringIO(temp), widths=[1] * n, header=None).to_numpy()
            p = np.shape(data)[1]
            out = data.reshape(1, (n*p))
            if i == 0:
                output = out
            else:
                output = np.vstack((output, out))
    return output

# compute distance w/ l2 norm

def distV(tr, tt):
    n_tr = tr.shape[0]
    n_tt = tt.shape[0]
    dist = np.zeros((n_tt, n_tr))
    # vectorized computation to form dist matrix
    dist = np.sqrt((tt**2).sum(axis=1)[:, np.newaxis] + (tr**2).sum(axis=1) - 2 * tt.dot(tr.T))
    return dist


# KNN classification (K = 10)

def knncls(tr, tt, k):
    # initialization
    train = b2vM(tr)
    n_tr = np.shape(train)[0]
    y_tr = getLabel(tr)
    test = b2vM(tt)
    n_tt = np.shape(test)[0]
    y_tt = getLabel(tt)

    # compute dist
    dist = distV(train, test)
    dist_id = dist.argsort()[:, :k]

    # get predicted labels
    y_tr_rep = np.tile(y_tr, (np.shape(dist_id)[0], 1))
    y_trf = np.take_along_axis(y_tr_rep, dist_id, axis=1)
    u, indices = np.unique(y_trf, return_inverse=True)
    label_fit = u[
        np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(y_trf.shape), None, np.max(indices) + 1), axis=1)]

    # compute/return prediction & error rate
    error_rate = 1 - sum(label_fit == y_tt) / n_tt
    return label_fit, error_rate


k = 10
result = knncls('trainingDigits', 'testDigits', k=k)
print("the error rate for "+str(k)+"-nearest neighbor is", result[1])

# n=5 fold cross validation with stratified sampling
# K arbitrarily choosen as 5, 10, 20, 30, 40, 50

k_cv = np.array([5, 10, 20, 30, 40, 50])
x_cv = b2vM('trainingDigits')
y_cv = getLabel('trainingDigits')

# Compute error rate for each fold/K

def knnclsCV(x, y, k_choice, n_fold):
    st = StratifiedKFold(n_splits=n_fold)
    n_k = len(k_choice)
    error_rate = np.zeros((n_fold, n_k))
    for j in range(n_k):
            k = k_choice[j]
            for i, (train_index, test_index) in enumerate(st.split(x, y)):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                dist = distV(x_train, x_test)
                dist_id = dist.argsort()[:, :k]
                y_tr_rep = np.tile(y_train, (np.shape(dist_id)[0], 1))
                y_trf = np.take_along_axis(y_tr_rep, dist_id, axis=1)
                u, indices = np.unique(y_trf, return_inverse=True)
                label_fit = u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(y_trf.shape), None, np.max(indices) + 1), axis=1)]
                error_rate[i, j] = 1 - sum(label_fit == y_test)/len(y_test)
                print("computing error rate for fold",str(i+1),"k ="+str(k_choice[j]))
    return error_rate

error_CV = knnclsCV(x_cv, y_cv, k_cv, 5)

best_k = error_CV.mean(axis = 0).argsort()
print("the optimal k =",str(k_cv[best_k][1]))

# error rate loosely bounded by double Bayes risk:
#(2BR - [Class/(1 - Class)]BR^2) when n --> inf (Cover & Hart 1967)
# massive adaptive knn methods were proposed, a more specific review check (G.H. Chen and D. Shah 2018)