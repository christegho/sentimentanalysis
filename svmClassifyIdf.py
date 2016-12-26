import sys
from tokenizeNgrams import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.neural_network import MLPClassifier

def svmClassifyIdf(posindir, negindir, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, stemmer, negation, ngram):

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

    vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True, ngram_range = (ngram,ngram))
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


    

    vsize = len(vectorizer.get_feature_names())
    mnbsize = len(classifierMNB.get_params(deep=True))
    svmsize = len(classifierSVM.get_params(deep=True))

    print("Vocabulary size")
    print vsize
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
    #print("Results for MLP")
    #print(classification_report(testLabels, predictionMLP))

    return tpSVM, tnSVM, tpNB, tnNB, 0, 0, predictionSVM, predictionMNB, [], vsize, mnbsize, svmsize