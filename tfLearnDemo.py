from getDocLabels import *
from splitDocs import *
from getSignificance import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)
nfold = 10
ngramMax = 3
maxIteration = 10

allGramsResults = {}
significantTestArrays = {}
vsize = {}


iteration = 0

trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)

import sys
from tokenizeNgrams import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from getDocLabels import *
from splitDocs import *
from getSignificance import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

nfold = 10
ngramMax = 3
maxIteration = 10



stemmer = False
negation = False

trainData = []
trainLabels = []
trainPosDocs.remove('cv622_8147.txt')
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




vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True, ngram_range = (1,3))
trainVectors = vectorizer.fit_transform(trainData)


testData = []
testLabels = []

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

testVectors = vectorizer.transform(testData)



from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb




# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainVectors, trainLabels, validation_set=(testVectors, testLabels), show_metric=True,
          batch_size=32)


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainVectors, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)