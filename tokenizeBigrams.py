import os
import re
from nltk import bigrams
# def tokenizeBigrams(indir):
# 	"""
# 	@return a list of filenames where each filename has a list of the words in it.
# 		example {'tets.txt': ['test', 'fun']}
# 	@author
# 	"""
	
# 	#indir = 'C:/Users/chris/Documents/sentimentanalysis/POS'
# 	#'/usr/groups/mphil/L90/data/POS'
# 	documents = {}
# 	for root, dirs, filenames in os.walk(indir):
# 		for filename in filenames:
# 			file = open(indir+'\\'+filename,'r')
# 			text = file.read().lower()    
# 			file.close()

# 			words = re.findall(r"[\w']+|[.,!?;]*", text)
# 			text = ' '.join(filter(None, words))
# 			bigramFeatureVector = []
#     		for item in bigrams(text):
#         	 bigramFeatureVector.append(' '.join(item))

# 			documents[filename]=bigramFeatureVector


# 		return documents


def tokenizeBigrams(sentence, grams):
    text = sentence.lower()
    words = re.findall(r"[\w']+|[.,!?;]*", text)
    words = filter(None, words)
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += [" ".join(words[i:i+gram])]
    return tokens


