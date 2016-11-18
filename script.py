from tokenize import *
from getUniqueWords import *
from getLexicon import *
from getSymbolicScore import *
from symScoreClassify import *
from splitData import *
from getFeatureVector import *
from naiveBayesClassify import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocs = tokenize(posindir)
negDocs = tokenize(negindir)

testPosDocs = posDocs
nfold = 10
posLexicon, negLexicon, posLexiconWeights, negLexiconWeights = getLexicon()

results = np.zeros((10,8))
for iteration in range(0,nfold):
	print iteration
	trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, nfold, iteration)
	resultsIteration = symScoreClassify(testPosDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, True)
	print resultsIteration[4:12]
	results[iteration,:] += np.array(resultsIteration[4:12])
	resultsIteration = symScoreClassify(testNegDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, False)
	print resultsIteration[4:12]
	results[iteration,:] += np.array(resultsIteration[4:12])

print np.matrix(results).mean(0)

#Naive Bayes
posVocabulary = getUniqueWords(trainPosDocs)
negVocabulary = getUniqueWords(trainNegDocs)

vocabulary = set(posVocabulary+negVocabulary)

results = np.zeros((10,4))
for iteration in range(0,nfold):
	print iteration
	trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, nfold, iteration)
	posDocF, negDocF = getFeatureVector(vocabulary, trainPosDocs, trainNegDocs, 1)
	posDocFSmoothing, negDocFSmoothing = getFeatureVector(vocabulary, trainPosDocs, trainNegDocs, 1)
	resultsIteration = naiveBayesClassify(testPosDocs, vocabulary, posDocF, negDocF, True, 1)
	print resultsIteration
	results[iteration,:] += np.array(resultsIteration)
	resultsIteration = naiveBayesClassify(testNegDocs, vocabulary, posDocF, negDocF, False, 1)
	print resultsIteration
	results[iteration,:] += np.array(resultsIteration)


results = np.zeros((10,4))
for iteration in range(0,nfold):
	print iteration
	trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, nfold, iteration)
	posDocF, negDocF = getFeatureVector(vocabulary, trainPosDocs, trainNegDocs, 0)
	posDocFSmoothing, negDocFSmoothing = getFeatureVector(vocabulary, trainPosDocs, trainNegDocs, 0)
	resultsIteration = naiveBayesClassify(testPosDocs, vocabulary, posDocF, negDocF, True, 0)
	print resultsIteration
	results[iteration,:] += np.array(resultsIteration)
	resultsIteration = naiveBayesClassify(testNegDocs, vocabulary, posDocF, negDocF, False, 0)
	print resultsIteration
	results[iteration,:] += np.array(resultsIteration)