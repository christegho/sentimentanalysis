def getSymbolicScore(lexicon, document):
	sLexicon = 0;	
	vectorLexicon = []
	weightedVecLex = []
	for word in lexicon:
		wordCount = document.count(word)
		if wordCount > 0:
			vectorLexicon.append(1)
			weightedVecLex.append(wordCount)
			sLexicon += 1

		else:
			vectorLexicon.append(0)
			weightedVecLex.append(0)

	return sLexicon, vectorLexicon, weightedVecLex



