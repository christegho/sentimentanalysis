from naiveBayesClassify2 import *
from buildDict import *
from tokenizeNgrams import *
from computeRatio import *
from processFiles import *

import numpy as np

def naiveBayes(posindir, negindir, ngram, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, alpha, stemmer, negation):
	poscounts  = buildDict(posindir, ngram, trainPosDocs, stemmer, negation)
	negcounts  = buildDict(negindir, ngram, trainNegDocs, stemmer, negation)

	vocabulary, ratio, posProbs, negProbs = computeRatio(poscounts, negcounts, alpha)
	posClassTestDocs = []
	negClassTestDocs = []
	print "Vocabulary Size:"
	print len(vocabulary)

	results = np.zeros((1,4))
	resultsNB = naiveBayesClassify2(testPosDocs, vocabulary, posProbs, negProbs, True, ngram, posindir, stemmer, negation)

	results += resultsNB[:4]
	posClassTestDocs += resultsNB[4]
	negClassTestDocs += resultsNB[5]

	resultsNB = naiveBayesClassify2(testNegDocs, vocabulary, posProbs, negProbs, False, ngram, negindir, stemmer, negation)
	results += resultsNB[:4]
	posClassTestDocs += resultsNB[4]
	negClassTestDocs += resultsNB[5]

	return results, posClassTestDocs, negClassTestDocs



