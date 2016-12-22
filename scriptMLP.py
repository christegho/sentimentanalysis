

trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, 10, 1)

posVocabulary = getUniqueWords(trainPosDocs)
negVocabulary = getUniqueWords(trainNegDocs)
vocabulary = set(posVocabulary+negVocabulary)

xTrainWeighted = [];
xTrain = [];
yTrain = [];
for doc in trainPosDocs:
 symbolicScores = getSymbolicScore(vocabulary, trainPosDocs[doc])
 xTrainWeighted.append(symbolicScores[2])
 xTrain.append(symbolicScores[1])
 yTrain.append(1)

for doc in trainNegDocs:
 symbolicScores = getSymbolicScore(vocabulary, trainNegDocs[doc])
 xTrainWeighted.append(symbolicScores[2])
 xTrain.append(symbolicScores[1])
 yTrain.append(0)

xTest = [];
xTestWeighted = []
for doc in testPosDocs:
 symbolicScores = getSymbolicScore(vocabulary, testPosDocs[doc])
 xTest.append(symbolicScores[1])
 xTestWeighted.append(symbolicScores[2])

for doc in testNegDocs:
 symbolicScores = getSymbolicScore(vocabulary, testNegDocs[doc])
 xTest.append(symbolicScores[1])
 xTestWeighted.append(symbolicScores[2])

#http://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier

#training and fitting of the MLP
clf = MLPClassifier(solver='lbfgs', alpha=.8, hidden_layer_sizes=(10, 6), random_state=1)
clf.fit(xTrain, yTrain) 

#predict labels
yTest = clf.predict(xTest) 

tp = sum(yTest[:99])
tn = sum(yTest[99:])	



#SVM
from sklearn import svm
clf = svm.SVC()
clf.fit(xTrain, yTrain) 

#predict labels
yTest = clf.predict(xTest) 