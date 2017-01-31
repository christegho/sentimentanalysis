from sklearn.neural_network import MLPClassifier
from buildDict import *
from computeRatio import *
from processFilesMLP import *
from processFilesMLP2 import *

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)
nfold = 3

iteration = 0
trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)

ngram= [2]
stemmer = True
negation = True
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
		 #xTrainWeighted.append(sparse.csr_matrix((np.ones(len(indecesWeighted)), indecesWeighted, [0,3])))
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


# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(sparse.csr_matrix(xTrainWeighted), yTrain)
t1 = time.time()
prediction_linear = classifier_linear.predict(sparse.csr_matrix(xTestWeighted)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

tp = sum(prediction_linear[:99])
fp = sum(prediction_linear[99:])
tn = 99-fp
fn = 99-tp	