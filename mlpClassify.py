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
	classifierMLP = MLPClassifier(solver='lbfgs', alpha=.1, hidden_layer_sizes=(10, 6), random_state=1)
	classifierMLP.fit(xTrainWeighted, yTrain) 

	#predict labels
	yTest = classifierMLP.predict(xTestWeighted) 

	tp = sum(yTest[:99])
	fp = sum(yTest[99:])
	tn = 99-fp
	fn = 99-tp	

	return tp, tn, fp, fn, yTest


# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(xTrainWeighted, yTrain)
t1 = time.time()
prediction_linear = classifier_linear.predict(xTestWeighted)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

tp = sum(yTest[:99])
fp = sum(yTest[99:])
tn = 99-fp
fn = 99-tp	