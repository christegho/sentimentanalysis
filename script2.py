from buildDict import *
from tokenizeNgrams import *
from computeRatio import *
from getDocLabels import *
from splitDocs import *
from processFiles import *
from naiveBayes import *
posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)


allGramsResults = {}
for n in range(3):
 ngram = [n+1]
 allGramsResults[ngram[0]] = np.zeros((10,4))

significantTestArrays = {}
nfold = 10
#for iteration in range(0,nfold):
iteration = 0
significantTestArrays = {}
significantTestArrays[ngram[0]] = [[],[]]
alpha=1
stemmer = True
negation = True
for n in range(3): 
 ngram = [n+1]
 trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)
 results = naiveBayes(posindir, negindir, ngram, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, alpha, stemmer, negation)
 print results[0]
 allGramsResults[ngram[0]][iteration,:] += results[0][0]
 significantTestArrays[ngram[0]] = [[],[]]
 significantTestArrays[ngram[0]][0] = results[1]
 significantTestArrays[ngram[0]][1] = results[2]


