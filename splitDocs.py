def splitDocs(posDocs, negDocs, nfold, iteration):

	posEnd = (len(posDocs)*(iteration)/nfold)-1;
	if (posEnd<0):
		posEnd = 0
	negEnd = (len(negDocs)*(iteration)/nfold)-1;
	if (negEnd<0):
		negEnd = 0
	trainPosDocs1 = posDocs[:posEnd]
	trainNegDocs1 = negDocs[:negEnd]



	testPosDocs = posDocs[int(len(posDocs)*(iteration)/nfold):int(len(posDocs)*(iteration+1)/nfold)-1]
	testNegDocs = negDocs[int(len(negDocs)*(iteration)/nfold):int(len(negDocs)*(iteration+1)/nfold)-1]


	posStart = len(posDocs)*(iteration+1)/nfold
	if (posStart > len(posDocs)):
		posStart = len(posDocs)
	negStart = len(negDocs)*(iteration+1)/nfold
	if (negStart > len(negDocs)):
		negStart = len(negDocs)

	trainPosDocs2 = posDocs[int(posStart):]
	trainNegDocs2 = negDocs[int(negStart):]

	trainPosDocs = trainPosDocs1 + trainPosDocs2
	trainNegDocs = trainNegDocs1 + trainNegDocs2

	return trainPosDocs, trainNegDocs, testPosDocs, testNegDocs


