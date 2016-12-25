from getDocLabels import *
from splitDocs import *
from svmClassify import *
from getSignificance import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocsLabels = getDocLabels(posindir)
negDocsLabels = getDocLabels(negindir)
nfold = 10
ngramMax = 3
maxIteration = 10

#vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True)

allGramsResults = {}
significantTestArrays = {}
vsize = {}

for classify in ['NBFF', 'SVMFF', 'NBTF', 'SVMTF', 'SVM', 'NB']:
 allGramsResults[classify] = {}
 vsize[classify] = {}
 for n in range(ngramMax):
  ngram = n+1
  allGramsResults[classify][ngram]= np.zeros((nfold,2))
  vsize[classify][ngram] = np.zeros((nfold,1))

for classify in ['NBFF', 'SVMFF', 'NBTF', 'SVMTF', 'SVM', 'NB']:
 significantTestArrays[classify] = {}
 for n in range(ngramMax):
  ngram = n+1
  significantTestArrays[classify][ngram] = np.zeros((10,198))


for iteration in range(0,maxIteration):
 print('iteration')
 trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitDocs(posDocsLabels, negDocsLabels, nfold, iteration)
 print iteration
 for n in range(ngramMax):
  ngram = n+1
  print('ngram')
  print ngram
  stemmer = False
  negation = False
  results = svmClassify(posindir, negindir, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, stemmer, negation, ngram)
  allGramsResults['SVMFF'][ngram][iteration,:] = results[:2]
  allGramsResults['NBFF'][ngram][iteration,:] = results[2:4]
  #allGramsResults['MLPFF'][ngram][iteration,:] += results[4:6]
  significantTestArrays['SVMFF'][ngram][iteration] = results[6]
  significantTestArrays['NBFF'][ngram][iteration] = results[7]
  #significantTestArrays['MLP'][ngram][iteration] = results[8]
  vsize['SVMFF'][ngram][iteration] = results[9]
  vsize['NBFF'][ngram][iteration] = results[9]
  stemmer = True
  negation = False
  results = svmClassify(posindir, negindir, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, stemmer, negation, ngram)
  allGramsResults['SVMTF'][ngram][iteration,:] = results[:2]
  allGramsResults['NBTF'][ngram][iteration,:] = results[2:4]
  #allGramsResults['MLPTF'][ngram][iteration,:] += results[4:6]
  significantTestArrays['SVMTF'][ngram][iteration] = results[6]
  significantTestArrays['NBTF'][ngram][iteration] = results[7]
  #significantTestArrays['MLP'][ngram][iteration] = results[8]
  vsize['SVMTF'][ngram][iteration] = results[9]
  vsize['NBTF'][ngram][iteration] = results[9]
  stemmer = True
  negation = True
  results = svmClassify(posindir, negindir, trainPosDocs, trainNegDocs, testPosDocs, testNegDocs, stemmer, negation, ngram)
  allGramsResults['SVM'][ngram][iteration,:] = results[:2]
  allGramsResults['NB'][ngram][iteration,:] = results[2:4]
  #allGramsResults['MLP'][ngram][iteration,:] += results[4:6]
  significantTestArrays['SVM'][ngram][iteration] = results[6]
  significantTestArrays['NB'][ngram][iteration] = results[7]
  #significantTestArrays['MLP'][ngram][iteration] = results[8]
  vsize['SVM'][ngram][iteration] = results[9]
  vsize['NB'][ngram][iteration] = results[9]
  #vsize['MLP'][ngram][iteration] = results[9] 

sigResults = getSignificance(significantTestArrays, nfold, ngramMax)

accResults = {}
for classify in ['NBFF', 'SVMFF', 'NBTF', 'SVMTF', 'SVM', 'NB']:
 accResults[classify] = {} 
 for n in range(ngramMax):
  ngram = n+1
  accResults[classify][ngram]= allGramsResults[classify][ngram].T.mean(1)
  

thefile = open('accResults.txt', 'w')
for classify in accResults: 
 for ngram in accResults[classify]:
  item = accResults[classify][ngram]
  thefile.write("%s\n" %classify)  
  thefile.write("%s\n" %ngram)
  thefile.write("%s\n" %item)

thefile.close()

thefile = open('sigResults.txt', 'w')
for item in sigResults:
  thefile.write("%s\n" %item)

thefile.close()

thefile = open('vsize.txt', 'w')
for classify in vsize: 
 for ngram in vsize[classify]:
  item = vsize[classify][ngram]
  thefile.write("%s\n" %classify)  
  thefile.write("%s\n" %ngram)
  thefile.write("%s\n" %item.mean())

thefile.close()