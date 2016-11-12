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
iteration = 1
trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, nfold, iteration)

posLexicon, negLexicon, posLexiconWeights, negLexiconWeights = getLexicon()

_, _, _, _, tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn = symScoreClassify(trainPosDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, True)
print tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn

_, _, _, _, tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn = symScoreClassify(trainNegDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, False)
print tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn

#Naive Bayes
posVocabulary = getUniqueWords(posDocs)
negVocabulary = getUniqueWords(negDocs)

vocabulary = set(posVocabulary+negVocabulary)

posDocF, negDocF = getFeatureVector(vocabulary, posDocs, negDocs, 1)
posDocFSmoothing, negDocFSmoothing = getFeatureVector(vocabulary, posDocs, negDocs, 1)

tp, tn, fp, fn = naiveBayesClassify(testPosDocs, vocabulary, posDocF, negDocF, True, 0)
print tp, tn, fp, fn

tp, tn, fp, fn = naiveBayesClassify(testNegDocs, vocabulary, posDocF, negDocF, False, 0)
print tp, tn, fp, fn

tp, tn, fp, fn = naiveBayesClassify(testPosDocs, vocabulary, posDocFSmoothing, negDocF, True, 1)
print tp, tn, fp, fn

tp, tn, fp, fn = naiveBayesClassify(testNegDocs, vocabulary, posDocFSmoothing, negDocF, False, 1)
print tp, tn, fp, fn


