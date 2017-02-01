import sys
from tokenizeNgrams import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from getDocLabels import *
from splitDocs import *
from svmClassifyIdf import *
from getSignificance import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

trainPosDocs = getDocLabels(posindir)
trainNegDocs = getDocLabels(negindir)

stemmer = False
negation = False
ngram = [1,2,3]

trainData = []
trainLabels = []

for fname in trainPosDocs:
    with open(posindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, ngram, stemmer, negation)
        trainData.append(' '.join(tokens))
        trainLabels.append(1)


for fname in trainNegDocs:
    with open(negindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, ngram, stemmer, negation)
        trainData.append(' '.join(tokens))
        trainLabels.append(0)


vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True, ngram_range = (1,3))
trainVectors = vectorizer.fit_transform(trainData)
        
classifierSVM = svm.LinearSVC()
classifierSVM.fit(trainVectors, trainLabels)

classifierMNB = MultinomialNB()
classifierMNB.fit(trainVectors, trainLabels)


posindirDemo = os.path.abspath('') + '\\POSDEMO'
negindirDemo = os.path.abspath('') + '\\NEGDEMO'

testPosDocs = getDocLabels(posindirDemo)
testNegDocs = getDocLabels(negindirDemo)
testData = []
testLabels = []

for fname in testPosDocs:
        with open(posindirDemo+ '/' + fname, 'r') as f:
            content = f.read()
            tokens = tokenizeNgrams(content, ngram, stemmer, negation)
            testData.append(' '.join(tokens))
            testLabels.append(1)

for fname in testNegDocs:
    with open(negindirDemo+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, ngram, stemmer, negation)
        testData.append(' '.join(tokens))
        testLabels.append(0)

testVectors = vectorizer.transform(testData)

predictionSVM = classifierSVM.predict(testVectors)
print(predictionSVM)
print(classifierSVM.decision_function(testVectors))


predictionMNB = classifierMNB.predict(testVectors)
print(predictionMNB)
print(classifierMNB.predict_proba(testVectors))

