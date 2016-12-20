from buildDict import *
from tokenizeBigrams import *
from compute_ratio import *
from getDocLabels import *
from splitDocs import *
from processFiles import *
from naiveBayesClassify2 import *
posindir = os.path.abspath('') + '\\POS1'
negindir = os.path.abspath('') + '\\NEG1'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)

trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, 3, 1)


ngram = [2]
poscounts  = buildDict(posindir, ngram, trainPosDocs)
negcounts  = buildDict(negindir, ngram, trainNegDocs)


vocabulary, r, posProbs, negProbs = compute_ratio(poscounts, negcounts, alpha=1)

#featureVectorIndexed, featureVector = processFiles(posindir+'/cv985_6359.txt', dic, r, ngram)

naiveBayesClassify2(testPosDocs, vocabulary, posProbs, negProbs, True, ngram, posindir)