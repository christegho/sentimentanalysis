import os
import re

def tokenize(indir):
	"""
	@return a list of filenames where each filename has a list of the words in it.
		example {'tets.txt': ['test', 'fun']}
	@author
	"""
	
	#indir = 'C:/Users/chris/Documents/sentimentanalysis/POS'
	#'/usr/groups/mphil/L90/data/POS'
	documents = {}
	for root, dirs, filenames in os.walk(indir):
	    for filename in filenames:
			file = open(indir+'\\'+filename,'r')
	    	text = file.read().lower()    
			file.close()
			words = re.findall(r"[\w']+|[.,!?;]*", text)
			documents[filename]=filter(None, words)

		return documents


