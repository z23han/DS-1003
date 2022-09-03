# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code

import numpy as np


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def convert(words):
    """
    Implements a method to convert a list of words into a sparse bag-of-words (dictionary representation)
    """
    from collections import Counter

    ct = Counter(words)
    return dict(ct)



def obj_loss_one(X_i, y_i, weights, lambd, m):
    return lambd/2 * np.dot(weights.T, weights) + max(0, 1 - y_i * np.dot(weights, X_i))


def obj_loss(X, y, weights, lambd):
    m = X.shape[0]
    loss = 0
    for i in range(m):
        loss += obj_loss_one(X[i], y[i], weights, lambd, m)
    return loss/m



def pegasos(X, y, weights=None, lambd=1, num_iter=1000):
    """
    Implements the pegasos algorithm on sparse data representation
    @param numpy.array X
    @param numpy.array y
    @param numpy.array weights
    @param lambd float scalar
    @param num_iter int scalar

    @return numpy array
    """
    if not weights:
        weights = np.zeros(X.shape[1])
    t = 0
    m = len(y)
    lst = np.arange(m)

    for i in range(num_iter):
        np.random.shuffle(lst)
        for j in lst:
            t = t + 1
            eta_t = 1./(t*lambd)
            if y[j]*np.dot(weights.T, X[j]) < 1:
                weights = (1 - eta_t*lambd) * weights + eta_t* np.dot(X[j].T, y[j])
            else:
                weights = (1 - eta_t*lambd) * weights
        if (i+1) % 10 == 0:
            print("iteration={}, loss={}".format(i+1, obj_loss(X, y, weights, lambd)))
    return weights


