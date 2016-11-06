def symScoreClassify(testDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights):
	#classify documents based on symbolic score and weighted symbolic score
	posClassWeightedSS = []
	posClassNonWeightedSS = []
	negClassWeightedSS = []
	negClassNonWeightedSS = []

	for document in testDocs:
		posSymbolicScores = getSymbolicScore(posLexicon, testDocs[document])
		negSymbolicScores = getSymbolicScore(negLexicon, testDocs[document])
		if (posSymbolicScores[0] > negSymbolicScores[0]):
			posClassNonWeightedSS.append(document)
		else:
			negClassNonWeightedSS.append(document)
		posWeightedScores = np.array(posSymbolicScores[1])*posLexiconWeights
		negWeightedScores = np.array(negSymbolicScores[1])*negLexiconWeights
		if (sum(posWeightedScores) > sum(negWeightedScores)):
			posClassWeightedSS.append(document)
		else:
			negClassWeightedSS.append(document)

	return posClassWeightedSS, posClassNonWeightedSS, negClassWeightedSS, negClassNonWeightedSS