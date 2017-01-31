from tokenizeNgrams import *

def processFiles(filename, vocabulary, ratio, ngram, stemmer, negation):
    indexesFile = []
    for doc in open(filename).xreadlines():
        tokens = tokenizeNgrams(doc, ngram, stemmer, negation)
        indexes = []
        for token in tokens:
            try:
                indexes += [vocabulary[token]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexesFile += indexes
        indexesFileUnsorted = indexesFile
        indexesFile.sort()
        featureVectorIndexed = []
        featureVector = []
        for i in indexesFile:
            featureVectorIndexed += ["%i:%f" % (i + 1, ratio[i])]
            featureVector.append(ratio[i])

    return featureVectorIndexed, featureVector