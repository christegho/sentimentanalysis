from getSymbolicScore import *

def getLexiconWeights(lexicon, documents, negdocuments, alpha) :
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