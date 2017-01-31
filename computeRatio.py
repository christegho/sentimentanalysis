import numpy as np
def computeRatio(poscounts, negcounts, alpha):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for token in alltokens:
        p[dic[token]] += poscounts[token]
        q[dic[token]] += negcounts[token]
    p /= abs(p).sum()
    q /= abs(q).sum()
    ratio= np.log(p/q)
    p = np.log(p)
    q = np.log(q)
    return dic, ratio, p, q 