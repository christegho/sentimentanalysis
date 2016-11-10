def getWordNotVocab(vocabulary, document):
	"""
	@return sLexicon: total number of words from lexicon or featur vector that appears in the document
	@return vectorLexicon: a vector with 1 if feature or word from lexicon appears at least once in document
	@return weightedVecLex: a vector with number of feature / word from lexicon in document i.e. document vector
	@author ct506 Chris Tegho
	"""
	wordsNotVocab = 0
	for word in document:
		wordCount = vocabulary.count(word)
		if wordCount == 0:
			wordsNotVocab += 1

	return wordsNotVocab
