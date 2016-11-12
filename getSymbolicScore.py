def getSymbolicScore(lexicon, document):
	"""
	@return sLexicon: total number of words from lexicon or featur vector that appears in the document
	@return vectorLexicon: a vector with 1 if feature or word from lexicon appears at least once in document
	@return weightedVecLex: a vector with number of feature / word from lexicon in document i.e. document vector
	@author ct506 Chris Tegho
	"""
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



