multiclass_tagger
=================
This algorithim ingests a set of documents and splits it into a training and test set. 
Using associated metadata--right now in the form of a .csv--the algorithm learns to apply multiple tags to a body of text.
Also, this can extract relevant entities from the text including people, places, organizations, etc.

This requires Python 2.7, Sci-Kit learn, NLTK, and NumPy. The tagging algorithm is a linear support vector classifier against a
tfidf of n-grams (3) set up as a one vs. rest classifier.
