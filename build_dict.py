import os
from tokenizeBigrams import *
from collections import Counter

def build_dict(indir, grams):
	#documents = {}
	dic = Counter()

	for root, dirs, filenames in os.walk(indir):
		for filename in filenames:
			for sentence in open(indir+'\\'+filename,'r').xreadlines():
				tokens = tokenizeBigrams(sentence, grams)
				dic.update(tokens)

			#documents[filename]=dic

	return dic