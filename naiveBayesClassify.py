from getSymbolicScore import *
from getWordNotVocab import *
import numpy as np

def naiveBayesClassify(testDocs, vocabulary, posDocF, negDocF, posClass, smoothing):
	"""
	Function classifies documents based on symbolic scores (wighted and non weighted) and returns confusion matrix values
	@author ct506 Chris Tegho
	"""

	tp = 0
	tn = 0
	fp = 0
	fn = 0

	for document in testDocs:
		symbolicScores = getSymbolicScore(vocabulary, testDocs[document]) 
		#wordsNotInVocab = getWordNotVocab(vocabulary, testDocs[document])

		posLik = sum(np.log10(np.array(symbolicScores[1])*posDocF+[1]*len(symbolicScores[1])-np.array(symbolicScores[1]))) #+sum(np.log10([smoothing]*wordsNotInVocab))
		negLik = sum(np.log10(np.array(symbolicScores[1])*negDocF+[1]*len(symbolicScores[1])-np.array(symbolicScores[1]))) #+sum(np.log10([smoothing]*wordsNotInVocab))
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