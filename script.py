from tokenize import *
from getUniqueWords import *
from getLexiconWeights import *
from getSymbolicScore import *
from getSymbolicScore import *
from splitData import *

import numpy as np

posindir = os.path.abspath('') + '\\POS'
negindir = os.path.abspath('') + '\\NEG'

posDocs = tokenize(posindir)
negDocs = tokenize(negindir)

testDocs = posDocs

posLexicon = getUniqueWords(posDocs)

negLexicon = getUniqueWords(negDocs)

posLexiconWeights = np.array(getLexiconWeights(posLexicon, posDocs, negDocs, 1))
negLexiconWeights = np.array(getLexiconWeights(negLexicon, negDocs, posDocs, 1))


symbolicClassification = symScoreClassify(testDocs, posLexicon, negLexicon, posLexiconWeights, negLexiconWeights)