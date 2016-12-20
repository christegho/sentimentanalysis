from processFiles import *
import numpy as np

def naiveBayesClassify2(testDocs, vocabulary, posProbs, negProbs, posClass, ngram, directory):
	"""
	@author ct506 Chris Tegho
	"""

	tp = 0
	tn = 0
	fp = 0
	fn = 0

	for document in testDocs:
		posLikIndexed, posLik = processFiles(directory + '/' + document, vocabulary, posProbs, ngram)
		negLikIndexed, negLik = processFiles(directory + '/' + document, vocabulary, negProbs, ngram)

		posLik = sum(posLik) #+sum(np.log10([smoothing]*wordsNotInVocab))
		negLik = sum(negLik) #+sum(np.log10([smoothing]*wordsNotInVocab))
		if (negLik == posLik):
			if (posClass):
				tp += .5
				fn += .5
			else:
				tn += .5
				fp += .5
		elif (posLik > negLik):
			if (posClass):
				tp += 1
			else:
				fp += 1
		else:
			if (posClass):
				fn += 1
			else:
				tn += 1

	return tp, tn, fp, fn