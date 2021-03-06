from getSymbolicScore import *
import numpy as np

def symScoreClassify(testDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, posClass):
	"""
	Function classifies documents based on symbolic scores (wighted and non weighted) and returns confusion matrix values
	@author ct506 Chris Tegho
	"""
	#classify documents based on symbolic score and weighted symbolic score
	posClassWeightedSS = [] #Vector with all docs classified as positive  by the weighted symbolic score
	posClassNonWeightedSS = [] #Vector with all docs classified as positive  by the non weighted symbolic score
	negClassWeightedSS = []
	negClassNonWeightedSS = []
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	weightedtp = 0
	weightedtn = 0
	weightedfp = 0
	weightedfn = 0
	classW = np.zeros((1,198))
	classNonW = np.zeros((1,198))
	i = 0
	for document in testDocs:
		posSymbolicScores = getSymbolicScore(posLexicon, testDocs[document]) #Get score for pos class lexicon
		negSymbolicScores = getSymbolicScore(negLexicon, testDocs[document]) #Get score for neg class lexicon
		if (posSymbolicScores[0] == negSymbolicScores[0]):
			posClassNonWeightedSS.append(document)
			negClassNonWeightedSS.append(document)
			if (posClass):
				tp += .5
				fn += .5
			else:
				tn += .5
				fp += .5
			if (i % 2 == 0):
				classNonW[0][i] = 1
		elif (posSymbolicScores[0] > negSymbolicScores[0]):
			posClassNonWeightedSS.append(document)
			classNonW[0][i] = 1
			if (posClass):
				tp += 1
			else:
				fp += 1
		else:
			negClassNonWeightedSS.append(document)
			if (posClass):
				fn += 1
			else:
				tn += 1

		posWeightedScores = np.array(posSymbolicScores[1])*posLexiconWeights
		negWeightedScores = np.array(negSymbolicScores[1])*negLexiconWeights
		if (sum(posWeightedScores) == sum(negWeightedScores)):
			posClassWeightedSS.append(document)
			negClassWeightedSS.append(document)
			if (posClass):
				weightedtp += .5
				weightedfn += .5
			else:
				weightedtn += .5
				weightedfp += .5
			if (i % 2 == 0):
				classW[0][i] = 1
		elif (sum(posWeightedScores) > sum(negWeightedScores)):
			posClassWeightedSS.append(document)
			if (posClass):
				weightedtp += 1
			else:
				weightedfp += 1
			classW[0][i] = 1
		else:
			negClassWeightedSS.append(document)
			if (posClass):
				weightedfn += 1
			else:
				weightedtn += 1
		i+=1

	return posClassWeightedSS, posClassNonWeightedSS, negClassWeightedSS, negClassNonWeightedSS, tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn, classW, classNonW