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


nfold = 10
posLexicon, negLexicon, posLexiconWeights, negLexiconWeights = getLexicon()
len(posLexicon), len(negLexicon), len(posLexiconWeights), len(negLexiconWeights)
resultsBow = np.zeros((10,8))
resultsSig2nonW = np.zeros((10,198))
resultsSig2W = np.zeros((10,198))

for iteration in range(0,nfold):
	 print iteration
	 trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, nfold, iteration)
	 resultsIteration = symScoreClassify(testPosDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, True)
	 print resultsIteration[4:12]
	 resultsBow[iteration,:] += np.array(resultsIteration[4:12])
	 resultsSig2nonW[iteration,:99] = resultsIteration[13][0][:99]
	 resultsSig2W[iteration,:99] = resultsIteration[12][0][:99]
	 resultsIteration = symScoreClassify(testNegDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, False)
	 print resultsIteration[4:12]
	 resultsBow[iteration,:] += np.array(resultsIteration[4:12])
	 resultsSig2nonW[iteration,99:] = resultsIteration[13][0][:99]
	 resultsSig2W[iteration,99:] = resultsIteration[12][0][:99]