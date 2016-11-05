def getLexiconWeights(lexicon, documents):
	lexiconWeights = [0]*length(lexicon)

	for document in documents:

		classSymbolicScores = getSymbolicScore(lexicon, documents[document]):
		negClassSymbolicScores = getSymbolicScore(lexicon, documents[document]):
		lexiconWeights = [i + j for i,j in zip(lexiconWeights, classSymbolicScores[3])]
		lexiconWeights = [i - j for i,j in zip(lexiconWeights, negClassSymbolicScores[3])]
	sumLexiconWeights = sum(lexiconWeights)
	if sumLexiconWeights != 0:
		lexiconWeights = [x / sumLexiconWeights for x in lexiconWeights]
	return lexiconWeights