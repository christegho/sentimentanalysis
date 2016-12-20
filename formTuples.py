def formTuples(docs, label):
	docTuples = [];
	for doc in docs:
		docTuples.append(tuple([docs[doc]]+ [label]))

	return docTuples