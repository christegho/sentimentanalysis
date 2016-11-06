from getSymbolicScore import *
import numpy as np

def symScoreClassify(testDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, posClass):
	#classify documents based on symbolic score and weighted symbolic score
	posClassWeightedSS = []
	posClassNonWeightedSS = []
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

	for document in testDocs:
		posSymbolicScores = getSymbolicScore(posLexicon, testDocs[document])
		negSymbolicScores = getSymbolicScore(negLexicon, testDocs[document])
		if (posSymbolicScores[0] == negSymbolicScores[0]):
			posClassNonWeightedSS.append(document)
			negClassNonWeightedSS.append(document)
			if (posClass):
				tp += .5
				fn += .5
			else:
				tn += .5
				fp += .5
		elif (posSymbolicScores[0] > negSymbolicScores[0]):
			posClassNonWeightedSS.append(document)
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
		elif (sum(posWeightedScores) > sum(negWeightedScores)):
			posClassWeightedSS.append(document)
			if (posClass):
				weightedtp += 1
			else:
				weightedfp += 1
		else:
			negClassWeightedSS.append(document)
			if (posClass):
				weightedfn += 1
			else:
				weightedtn += 1

	return posClassWeightedSS, posClassNonWeightedSS, negClassWeightedSS, negClassNonWeightedSS, tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn