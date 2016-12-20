from tokenizeBigrams import *
from tokenizeBigrams import *

def processFiles(filename, dic, r, ngram):
    indexesFile = []
    for l in open(filename).xreadlines():
        tokens = tokenizeBigrams(l, ngram)
        indexes = []
        for t in tokens:
            try:
                indexes += [dic[t]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexesFile += indexes
        indexesFile.sort()
        featureVectorIndexed = []
        featureVector = []
        for i in indexesFile:
            featureVectorIndexed += ["%i:%f" % (i + 1, r[i])]
            featureVector.append(r[i])

    return featureVectorIndexed, featureVector