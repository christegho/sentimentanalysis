def splitData(posDocs, negDocs, nfold, iteration):

	posEnd = (len(posDocs)*(iteration)/nfold)-1;
	if (posEnd<0):
		posEnd = 0
	negEnd = (len(negDocs)*(iteration)/nfold)-1;
	if (negEnd<0):
		negEnd = 0
	trainPosDocs1 = dict(posDocs.items()[:posEnd])
	trainNegDocs1 = dict(negDocs.items()[:negEnd])



	testPosDocs = dict(posDocs.items()[(len(posDocs)*(iteration)/nfold):(len(posDocs)*(iteration+1)/nfold)-1])
	testNegDocs = dict(negDocs.items()[(len(negDocs)*(iteration)/nfold):(len(negDocs)*(iteration+1)/nfold)-1])


	posStart = len(posDocs)*(iteration+1)/nfold
	if (posStart > len(posDocs)):
		posStart = len(posDocs)
	negStart = len(negDocs)*(iteration+1)/nfold
	if (negStart > len(negDocs)):
		negStart = len(negDocs)

	trainPosDocs2 = dict(posDocs.items()[posStart:])
	trainNegDocs2 = dict(posDocs.items()[negStart:])

	trainPosDocs = dict(trainPosDocs1.items() + trainPosDocs2.items())
	trainNegDocs = dict(trainNegDocs1.items() + trainNegDocs2.items())

	return trainPosDocs, trainNegDocs, testPosDocs, testNegDocs


