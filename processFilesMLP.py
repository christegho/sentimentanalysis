from tokenizeNgrams import *

def processFilesMLP(filename, vocabulary, ratio, ngram, stemmer, negation):
    indexesFile = []
    for doc in open(filename).xreadlines():
        tokens = tokenizeNgrams(doc, ngram, stemmer, negation)
        indexes = [0]*len(vocabulary)
        indexesWeighted = [0]*len(vocabulary)
        for token in tokens:
            try:
                indexes[vocabulary[token]] = 1
                indexesWeighted[vocabulary[token]] += 1
            except KeyError:
                pass
        
    return indexes, indexesWeighted