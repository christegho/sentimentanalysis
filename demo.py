import sys
from tokenizeNgrams import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from getDocLabels import *
from splitDocs import *
from svmClassify import *
from getSignificance import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posindirDemo = os.path.abspath('') + '\\POSDEMO'
negindirDemo = os.path.abspath('') + '\\NEGDEMO'


posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)

testPosDocs = getDocLabels(posindirDemo)
testNegDocs = getDocLabels(negindirDemo)

nfold = 10
ngramMax = 3
maxIteration = 10



stemmer = False
negation = False

trainData = []
trainLabels = []
testData = []
testLabels = []
for fname in trainPosDocs:
    with open(posindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1,2,3], stemmer, negation)
        trainData.append(' '.join(tokens))
        trainLabels.append(1)


for fname in trainNegDocs:
    with open(negindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1,2,3], stemmer, negation)
        trainData.append(' '.join(tokens))
        trainLabels.append(0)


for fname in testPosDocs:
        with open(posindirDemo+ '/' + fname, 'r') as f:
            content = f.read()
            tokens = tokenizeNgrams(content, [1], stemmer, negation)
            testData.append(' '.join(tokens))
            testLabels.append(1)


for fname in testNegDocs:
    with open(negindirDemo+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1], stemmer, negation)
        testData.append(' '.join(tokens))
        testLabels.append(0)

vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True, ngram_range = (1,3))
trainVectors = vectorizer.fit_transform(trainData)
testVectors = vectorizer.transform(testData)

classifierSVM = svm.LinearSVC()
classifierSVM.fit(trainVectors, trainLabels)
predictionSVM = classifierSVM.predict(testVectors)

classifierMNB = MultinomialNB()
classifierMNB.fit(trainVectors, trainLabels)
predictionMNB = classifierMNB.predict(testVectors)