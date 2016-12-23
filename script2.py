from buildDict import *
from tokenizeNgrams import *
from computeRatio import *
from getDocLabels import *
from splitDocs import *
from processFiles import *
from naiveBayes import *
from mlpClassify import *

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)


allGramsResultsNB = {}

for n in range(3):
 ngram = [n+1]
 allGramsResultsNB[ngram[0]] = np.zeros((10,4))

allGramsResultsMLP = allGramsResultsNB

significantTestArraysNB = {}
nfold = 3
#for iteration in range(0,nfold):
iteration = 0

alpha=1
stemmer = True
negation = True
for n in range(3): 
 ngram = [n+1]
 trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)
 
 naiveBayesResults = naiveBayes(posindir, negindir, ngram, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, alpha, stemmer, negation)
 print naiveBayesResults[0]
 allGramsResultsNB[ngram[0]][iteration,:] += naiveBayesResults[0][0]
 significantTestArraysNB[ngram[0]] = [[[]*nfold],[[]*nfold]]
 significantTestArraysNB[ngram[0]][0][iteration] = naiveBayesResults[1]
 significantTestArraysNB[ngram[0]][1][iteration] = naiveBayesResults[2]


