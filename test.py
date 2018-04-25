import numpy as np 

a = np.round(1000*np.random.normal(size=(1,10), loc=0.0, scale=10.0))
a = a.tolist()

def entropy2(labels):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)

    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        ent -= i * log(i, base=n_classes)

    return ent

entropy2(a)


