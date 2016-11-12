import numpy as np
import os

def getLexicon():
	indir = os.path.abspath('') + '\\sentlex' #/usr/groups/mphil/L90/resources/sent_lexicon'	

	posLexicon = []
	negLexicon = []
	posLexiconWeights = []
	negLexiconWeights = []


	with open(indir) as f:
	    lines = f.readlines()

	for line in lines:
		words = line.split()
		lexiconword = words[2].split('=')[1]
		polarity = words[5].split('=')[1]
		weight = words[0].split('=')[1]
		if (weight == 'strongsubj'):
			weight = 2 
		else:
			weight = 1
		if (polarity == 'negative'):
			if (not lexiconword in negLexicon):
				negLexicon.append(lexiconword) 
				negLexiconWeights.append(weight)
		else:
			if (not lexiconword in posLexicon):
				posLexicon.append(lexiconword) 
				posLexiconWeights.append(weight)

	posLexiconWeights = np.array(posLexiconWeights)
	negLexiconWeights = np.array(negLexiconWeights)

	return 	posLexicon, negLexicon, posLexiconWeights, negLexiconWeights
