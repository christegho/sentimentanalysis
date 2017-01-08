# sentimentanalysis

This project concerns sentiment classification of movie reviews using a machine learning
approach based on bag-of-word features, using a sentiment lexicon, a stemmer, a POS-tagger and
a syntactic parser. Extentions will be added as the project progresses.

1) The script scriptSYMB implements the baseline concerning the symbolic score and weighted sympolic score. The following functions were built for this purpose:
tokenize
splitData
symScoreClassify
getSymbolicScore

2) The script scriptNB implements the Naive Bayes algorithm with smoothing, with n-grams. The following functions were built for this purpose:
getDocLabels
splitDocs
naiveBayes
buildDict
computeRatio
naiveBayesClassify2
processFiles
tokenizeNgrams

3) The script scriptExtension implements SVMs with n-grams. The following functions were built for this purpose:
getDocLabels
splitDocs
svmClassify
svmClassifyIdf
getSignificance
tokenizeNgrams