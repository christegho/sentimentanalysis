from scipy.stats import binom_test
import numpy as np

def getSignificance(significantTestArrays, nfold, ngramMax):
	sigResults = []
	for classifier in significantTestArrays:
	 for classifier2 in significantTestArrays:
	  for n in range(ngramMax):
	   for n2 in range(ngramMax):
	    if (classifier != classifier2 or n != n2):
	     ngram = n + 1
	     ngram2 = n2 + 1
	     sys1 = classifier + str(ngram)
	     sys2 = classifier + str(ngram2)
	     iterResults = np.zeros([10,2])
	     for iter in range(nfold):
	      for i in range(198):
	       r1 = significantTestArrays[classifier][ngram][iter][i]
	       r2 = significantTestArrays[classifier2][ngram2][iter][i]
	       if (r1 == r2):
	        if ((i < 99 and r1 or 1) or (i >= 99 and r1 == 0)):
	         iterResults[iter,0] += .5
	         iterResults[iter,1] += .5
	       else:
	        if ((i < 99 and r1 == 1) or (i >= 99 and r1 == 0)):
	         iterResults[iter,0] += 1
	        if ((i < 99 and r2 == 1) or (i >= 99 and r2 == 0)):
	         iterResults[iter,1] += 1
	     sigResults.append({'sys1':sys1, 'sys2':sys2, 'sig' : binom_test(iterResults.T.mean(1))})

	return sigResults