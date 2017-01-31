import os
from tokenizeNgrams import *
from collections import Counter

def buildDict(indir, grams, docs, stemmer, negation):
	dic = Counter()

	for root, dirs, filenames in os.walk(indir):
		for filename in docs:
			for sentence in open(indir+'\\'+filename,'r').xreadlines():
				tokens = tokenizeNgrams(sentence, grams, stemmer, negation)
				dic.update(tokens)


	return dic