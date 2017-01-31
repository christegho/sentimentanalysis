from getSymbolicScore import *

def getFeatureVector(vocabulary, documents, negdocuments, smoothing) :
	"""
	Function returns weights for lexicon words based on number of times a word appears in a training document for a specific class. There is an option to reduce the weight if the word appears in the
	opposite class documents.
	Weights are normalized.
	@author ct506 Chris Tegho
	"""
	posDocF = [0]*len(vocabulary)
	negDocF = [0]*len(vocabulary)

	for document in documents:
		classSymbolicScores = getSymbolicScore(vocabulary, documents[document])
		posDocF = [i + j for i,j in zip(posDocF, classSymbolicScores[2])]

	for document in negdocuments:
		negClassSymbolicScores = getSymbolicScore(vocabulary, negdocuments[document])
		negDocF = [i + j for i,j in zip(negDocF, negClassSymbolicScores[2])]

	sumLexiconWeights = sum(posDocF)
	if sumLexiconWeights != 0:
		posDocF = [(float(x)+smoothing)/(sumLexiconWeights+smoothing*len(vocabulary)) for x in posDocF]

	sumNegLexiconWeights = sum(negDocF)
	if sumNegLexiconWeights != 0:
		negDocF = [(float(x)+smoothing)/(sumNegLexiconWeights+smoothing*len(vocabulary))  for x in negDocF]
	return posDocF, negDocF