from getSymbolicScore import *

def getLexiconWeights(lexicon, documents, negdocuments, alpha) :
	"""
	Function returns weights for lexicon words based on number of times a word appears in a training document for a specific class. There is an option to reduce the weight if the word appears in the
	opposite class documents.
	Weights are normalized.
	@author ct506 Chris Tegho
	"""
	lexiconWeights = [0]*len(lexicon)

	for document in documents:
		classSymbolicScores = getSymbolicScore(lexicon, documents[document])
		lexiconWeights = [i + j for i,j in zip(lexiconWeights, classSymbolicScores[2])]

	for document in negdocuments:
		negClassSymbolicScores = getSymbolicScore(lexicon, negdocuments[document])
		lexiconWeights = [i - alpha*j for i,j in zip(lexiconWeights, negClassSymbolicScores[2])]

	sumLexiconWeights = sum(lexiconWeights)
	if sumLexiconWeights != 0:
		lexiconWeights = [float(x)/sumLexiconWeights for x in lexiconWeights]
	return lexiconWeights