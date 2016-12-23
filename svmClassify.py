import sys
from tokenizeNgrams import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

def svmClassify(posindir, negindir, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, stemmer, negation, ngram):

    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    for fname in trainPosDocs:
        with open(posindir+ '/' + fname, 'r') as f:
            content = f.read()
            tokens = tokenizeNgrams(content, [1], stemmer, negation)
            trainData.append(' '.join(tokens))
            trainLabels.append(1)


    for fname in trainNegDocs:
        with open(negindir+ '/' + fname, 'r') as f:
            content = f.read()
            tokens = tokenizeNgrams(content, [1], stemmer, negation)
            trainData.append(' '.join(tokens))
            trainLabels.append(0)

    for fname in testPosDocs:
        with open(posindir+ '/' + fname, 'r') as f:
            content = f.read()
            tokens = tokenizeNgrams(content, [1], stemmer, negation)
            testData.append(' '.join(tokens))
            testLabels.append(1)


    for fname in testNegDocs:
        with open(negindir+ '/' + fname, 'r') as f:
            content = f.read()
            tokens = tokenizeNgrams(content, [1], stemmer, negation)
            testData.append(' '.join(tokens))
            testLabels.append(0)

    vectorizer = CountVectorizer(ngram_range = (ngram,ngram))
    trainVectors = vectorizer.fit_transform(trainData)
    testVectors = vectorizer.transform(testData)


    # Perform classification with SVM, kernel=linear
    classifierSVM = svm.LinearSVC()
    classifierSVM.fit(trainVectors, trainLabels)
    predictionSVM = classifierSVM.predict(testVectors)

    tpSVM = sum(predictionSVM[:99])
    fpSVM = sum(predictionSVM[99:])
    tnSVM = 99-fpSVM
    fnSVM = 99-tpSVM

    classifierMNB = MultinomialNB()
    classifierMNB.fit(trainVectors, trainLabels)
    predictionMNB = classifierMNB.predict(testVectors)

    tpNB = sum(predictionMNB[:99])
    fpNB =  sum(predictionMNB[99:])
    tnNB = 99-fpNB
    fnNB = 99-tpNB


    #training and fitting of the MLP
    classifierMLP = MLPClassifier(solver='lbfgs', alpha=.8, hidden_layer_sizes=(5, 4), random_state=1)
    classifierMLP.fit(trainVectors, trainLabels) 

    #predict labels
    predictionMLP = classifierMLP.predict(testVectors) 

    tpMLP = sum(predictionMLP[:99])
    tnMLP = sum(predictionMLP[99:])    
    tnMLP = 99-fpMLP
    fnMLP = 99-tpMLP

    vsize = len(vectorizer.get_params())
    mnbsize = len(classifierMNB.get_params())
    svmsize = len(classifierSVM.get_params())

    print("Vocabulary size")
    print vsize
    print("ngram")
    print ngram
    print("iteration")
    print iteration
    print("Results for SVM")
    print(classification_report(testLabels, predictionSVM))
    print svmsize
    print "TP and TN"
    print tpSVM, tnSVM
    print("Results for MNB")
    print(classification_report(testLabels, predictionMNB))
    print mnbsize
    print "TP and TN"
    print tpNB, tnNB
    print("Results for MLP")
    print(classification_report(testLabels, predictionMLP))

    return tpSVM, tnSVM, tpNB, tnNB, tpMLP, tnMLP, predictionSVM, predictionMNB, predictionMLP, vsize, mnbsize, svmsize