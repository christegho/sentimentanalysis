import os
from tokenizeBigrams import *
from collections import Counter

def buildDict(indir, grams, docs):
	dic = Counter()

	for root, dirs, filenames in os.walk(indir):
		for filename in docs:
			for sentence in open(indir+'\\'+filename,'r').xreadlines():
				tokens = tokenizeBigrams(sentence, grams)
				dic.update(tokens)


	return dic