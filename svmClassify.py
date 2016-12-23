import sys
import os
import time

from buildDict import *
from tokenizeNgrams import *
from computeRatio import *
from getDocLabels import *
from splitDocs import *
from processFiles import *
from naiveBayes import *
from mlpClassify import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)
nfold = 10

iteration = 0
trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)


stemmer = True
negation = True
#vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True)

train_data = []
train_labels = []
test_data = []
test_labels = []
for fname in trainPosDocs:
    with open(posindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1], stemmer, negation)
        train_data.append(' '.join(tokens))
        train_labels.append(1)


for fname in trainNegDocs:
    with open(negindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1], stemmer, negation)
        train_data.append(' '.join(tokens))
        train_labels.append(0)

for fname in testPosDocs:
    with open(posindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1], stemmer, negation)
        test_data.append(' '.join(tokens))
        test_labels.append(1)


for fname in testNegDocs:
    with open(negindir+ '/' + fname, 'r') as f:
        content = f.read()
        tokens = tokenizeNgrams(content, [1], stemmer, negation)
        test_data.append(' '.join(tokens))
        test_labels.append(0)

vectorizer = CountVectorizer(ngram_range = (2,2))
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)


# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

tpSVM = sum(prediction_liblinear[:99])
fpSVM = sum(prediction_liblinear[99:])
tnSVM = 99-fpSVM
fnSVM = 99-tpSVM

classifier_mnb = MultinomialNB()
classifier_mnb.fit(train_vectors, train_labels)
prediction_mnb = classifier_mnb.predict(test_vectors)

tpNB = sum(prediction_mnb[:99])
fpNB =  sum(prediction_mnb[99:])
tnNB = 99-fpNB
fnNB = 99-tpNB

from sklearn.neural_network import MLPClassifier

#training and fitting of the MLP
clf = MLPClassifier(solver='lbfgs', alpha=.8, hidden_layer_sizes=(5, 4), random_state=1)
clf.fit(train_vectors, train_labels) 

#predict labels
yTest = clf.predict(test_vectors) 

tp = sum(yTest[:99])
tn = sum(yTest[99:])    


print("Results for SVM")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))
print "TP and TN"
print tpSVM, tnSVM
print("Results for MNB")
print(classification_report(test_labels, prediction_mnb))
print "TP and TN"
print tpNB, tnNB
print("Results for MLP")
print(classification_report(test_labels, yTest))