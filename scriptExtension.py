from getDocLabels import *
from splitDocs import *
from svmClassify import *

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)
nfold = 10


stemmer = True
negation = True
#vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True)



allGramsResults = {}
significantTestArrays = {}
vsize = {}
nfold = 10

for classify in ['SVM', 'NB', 'MLP']
 allGramsResults[classify] = {}
 significantTestArrays[classify] = {}
 vsize[classify] = {}
 for n in range(3):
  ngram = n+1
  allGramsResults[classify][ngram]= np.zeros((10,4))
  significantTestArrays[classify][ngram] = np.zeros((10,198))
  vsize[classify][ngram] = np.zeros((10,1))

iteration = 0
trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)

for iteration in range(0,2):
 print iteration
 for n in range(2)
  print n
  ngram = n+1
  results = svmClassify(posindir, negindir, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, stemmer, negation, ngram)
  allGramsResultsp['SVM'][ngram][iteration,:] += results[:1]
  allGramsResultsp['NB'][ngram][iteration,:] += results[1:3]
  allGramsResultsp['MLP'][ngram][iteration,:] += results[3:5]
  significantTestArrays['SVM'][ngram][iteration] = results[6]
  significantTestArrays['NB'][ngram][iteration] = results[7]
  significantTestArrays['MLP'][ngram][iteration] = results[8]
  vsize['SVM'][ngram][iteration] = results[11]
  vsize['NB'][ngram][iteration] = results[10]
  vsize['MLP'][ngram][iteration] = results[9] 
