from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDB Dataset loading
train1, test1, actTest = imdb.load_data(path='own.pkl',
                                valid_portion=0.05)
trainX, trainY = train
testX, testY = test
actTestX, actTestY = actTest
# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
actTestX = pad_sequences(actTestX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
actTestY = to_categorical(actTestY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)

predictions = model.predict(actTestX)



testO, _ = imdb.load_data(path='owntest.pkl',
                                valid_portion=0.0)

trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

predictions = model.predict(actTestX)
tf = 0
tn = 0
for i in range(len(predictions)):
	if (predictions[i][0]>0.5 and actTestY[i][0] == 1):# or ( predictions[i][1]>0.5 and testY[i][1] == 1):
			tf+=1
	
	if (predictions[i][0]<0.5 and actTestY[i][0] == 0):# or ( predictions[i][1]<0.5 and testY[i][1] == 0):
			tn+=1