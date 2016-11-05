from tokenize import *
from getUniqueWords import *

indir = os.path.abspath('') + '\\POS'

posDocs = tokenize(indir)
posLexicon = getUniqueWords(posDocs)
posLexiconWeights = getLexiconWeights(posLexicon, posDocs)

for document in posDocs:

	posSymbolicScores = getSymbolicScore(posLexicon, posDocs[document]):
