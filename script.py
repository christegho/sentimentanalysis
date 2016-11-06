from tokenize import *
from getUniqueWords import *
from getLexiconWeights import *
from getSymbolicScore import *
from symScoreClassify import *
from splitData import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocs = tokenize(posindir)
negDocs = tokenize(negindir)

testPosDocs = posDocs
# trainPosDocs, trainNegDocs, testPosDocs, testNegDocs = splitData(posDocs, negDocs, nfold, iteration)

posLexicon = getUniqueWords(posDocs)
negLexicon = getUniqueWords(negDocs)

posLexiconWeights = np.array(getLexiconWeights(posLexicon, posDocs, negDocs, 1))
negLexiconWeights = np.array(getLexiconWeights(negLexicon, negDocs, posDocs, 1))


posClassWeightedSS, posClassNonWeightedSS, negClassWeightedSS, negClassNonWeightedSS, tp, tn, fp, fn, weightedtp, weightedtn, weightedfp, weightedfn = symScoreClassify(testPosDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights, True)