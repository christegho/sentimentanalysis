import os

def getDocLabels(indir):

	documents = []
	for root, dirs, filenames in os.walk(indir):
		for filename in filenames:
			documents.append(filename)

	return documents


