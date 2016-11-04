def getUniqueWords(documents) :
    uniqueWords = [] 
    for doc in documents:
    	for word in documents[doc]:
        	if not word in uniqueWords:
            	uniqueWords.append(word)
    return uniqueWords