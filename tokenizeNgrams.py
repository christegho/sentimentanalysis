import os
import re
from nltk import bigrams
from nltk.stem.porter import *
from nltk.sentiment.util import *

def tokenizeNgrams(sentence, grams, stemmer, negation):
	text = sentence.lower()
	words = re.findall(r"[\w']+|[(){}.,!?;]*", text)
	words = filter(None, words)
	if (stemmer):
		#Create a new Porter stemmer.
		stemmer = PorterStemmer()
		words = [stemmer.stem(word) for word in words]
	if (negation):
		words = mark_negation(words)
	tokens = []
	for gram in grams:
		for i in range(len(words) - gram + 1):
			tokens += [" ".join(words[i:i+gram])]
	return tokens


