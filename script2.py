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


allGramsResultsNBF = {}
nfold = 10
significantTestArraysNBF = {}

for n in range(3):
 ngram = [n+1]
 allGramsResultsNBF[ngram[0]] = np.zeros((10,4))
 significantTestArraysNBF[ngram[0]] = [[[1]*nfold],[[1]*nfold]]


for iteration in range(0,nfold):
 alpha=1
 stemmer = False
 negation = False
 trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)
 for n in range(3): 
  ngram = [n+1]
  naiveBayesResults = naiveBayes(posindir, negindir, ngram, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, alpha, stemmer, negation)
  print n
  print iteration
  print naiveBayesResults[0]
  allGramsResultsNBF[ngram[0]][iteration,:] += naiveBayesResults[0][0]
  significantTestArraysNBF[ngram[0]][0][0][iteration] = naiveBayesResults[1]
  significantTestArraysNBF[ngram[0]][1][0][iteration] = naiveBayesResults[2]


