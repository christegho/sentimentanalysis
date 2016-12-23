from sklearn.neural_network import MLPClassifier
from buildDict import *
from computeRatio import *
from processFilesMLP import *

def mlpClassify(posindir, negindir, ngram, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, alpha, stemmer, negation):
	poscounts  = buildDict(posindir, ngram, trainPosDocs, stemmer, negation)
	negcounts  = buildDict(negindir, ngram, trainNegDocs, stemmer, negation)

	vocabulary, ratio, posProbs, negProbs = computeRatio(poscounts, negcounts, alpha)

	xTrain = []
	xTrainWeighted = []
	yTrain = []
	for document in trainPosDocs:
		indeces, indecesWeighted = processFilesMLP(posindir + '/' + document, vocabulary, posProbs, ngram, stemmer, negation)
		#negLikIndexed, negLik = processFiles(directory + '/' + document, vocabulary, negProbs, ngram, stemmer, negation)
		xTrain.append(indeces)
		xTrainWeighted.append(indecesWeighted)
		yTrain.append(1)

	for document in trainNegDocs:
		indeces, indecesWeighted = processFilesMLP(negindir + '/' + document, vocabulary, negProbs, ngram, stemmer, negation)
		#negLikIndexed, negLik = processFiles(directory + '/' + document, vocabulary, negProbs, ngram, stemmer, negation)
		xTrain.append(indeces)
		xTrainWeighted.append(indecesWeighted)
		yTrain.append(0)


	xTest = [];
	xTestWeighted = [];
	for document in testPosDocs:
		indeces, indecesWeighted = processFilesMLP(posindir + '/' + document, vocabulary, posProbs, ngram, stemmer, negation)
		#negLikIndexed, negLik = processFiles(directory + '/' + document, vocabulary, negProbs, ngram, stemmer, negation)
		xTest.append(indeces)
		xTestWeighted.append(indecesWeighted)

	for document in testNegDocs:
		indeces, indecesWeighted = processFilesMLP(negindir + '/' + document, vocabulary, negProbs, ngram, stemmer, negation)
		#negLikIndexed, negLik = processFiles(directory + '/' + document, vocabulary, negProbs, ngram, stemmer, negation)
		xTest.append(indeces)
		xTestWeighted.append(indecesWeighted)

	#http://scikit-learn.org/stable/modules/neural_networks_supervised.html



	#training and fitting of the MLP
	clf = MLPClassifier(solver='lbfgs', alpha=.1, hidden_layer_sizes=(10, 6), random_state=1)
	clf.fit(xTrainWeighted, yTrain) 

	#predict labels
	yTest = clf.predict(xTestWeighted) 

	tp = sum(yTest[:99])
	fp = sum(yTest[99:])
	tn = 99-fp
	fn = 99-tp	

	return tp, tn, fp, fn, yTest