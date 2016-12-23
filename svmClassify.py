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
posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)
nfold = 10

iteration = 0
trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)


vectorizer = CountVectorizer(ngram_range = (1,1))
stemmer = False
negation = False
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


train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

tp = sum(prediction_liblinear[:99])
fp = sum(prediction_liblinear[99:])
tn = 99-fp
fn = 99-tp



# Print results in a nice table
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_labels, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))