
# coding: utf-8

# #Multi-Label Document Classification
# 
# This project aims to use 60 documents to create a multi-label classification system, with entity extraction as a fringe benefit. 

# In[287]:

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import random


# In[288]:

metadata = pd.read_csv("DocumentTagResults.csv")


# In[289]:

metadata.columns


# In[290]:

metadata['FileName'] = metadata['FileName'].replace(to_replace='.docx', value='.htm', regex=True)


# In[291]:

filenames = os.listdir('/Users/brandomr/Sites/docs')
#gets list of filenames in the document directory

len(filenames)
#tests how many files are in the docs folder

filenames = sorted(filenames)


# In[292]:

metadata = metadata.sort(columns = 'FileName')


# In[293]:

#this is a quick check on how many tags each document has
numbertags = metadata.pivot_table(values='DocumentID', rows=['FileName'], aggfunc = len)

print 'Per document the'
print 'max number of tags is ' + str(numbertags.max())
print 'min number of tags is ' + str(numbertags.min())
print 'mean number of tags is ' + str(numbertags.mean())
print 'median number of tags is ' + str(numbertags.median())


# In[299]:

#creates list of tuples for document tags
taglist = []
for i in range(len(filenames)):
    doctags = metadata[metadata['FileName'] == filenames[i]]['TagName'].tolist()
    doctags = sorted(doctags)
    taglist.append(doctags)

#this function pulls a random subset from a list
def random_subset(iterator, k):
    result = []
    n = 0
    
    for item in iterator:
        n +=1
        if len(result) < k:
            result.append(item)
        else:
            s = int(random.random()*n)
            if s < k: 
                result[s] = item
    
    return result

#the below code pulls 2 random tags and from the tag list per document    
#taglist_reduced = []
#for i in range(len(taglist)):
#    tagitem = random_subset(taglist[i],2)
#    taglist_reduced.append(tagitem)


# In[306]:

taglist_reduced[:10]


# In[300]:

corpora = []

for i in filenames:
    doc = open('/Users/brandomr/Sites/docs/'+ i)
    text = doc.read()
    #grabs the document as variable text
    
    text = nltk.clean_html(text)
    #strips html formatting
    
    text = text.replace('&#xa0;', '\xA0')
    text = text.decode('utf-8', 'ignore')
    #gets rid of non-break space html and converts to unicode
    
    corpora.append(text)
    #adds to corpora


# In[301]:

#tokenizes and chunks for entity extraction
def extract_entities(text):
    entities = []
    for sentence in nltk.sent_tokenize(text):
        #tokenizes into sentence
        
        chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
        #tokenizes into words, then tags by parts of speech, then uses nltk's built in chunker
        
        entities.extend([chunk for chunk in chunks if hasattr(chunk, 'node')])
        #iterates through the text and pulls any chunks that are nodes--these are the entities
        
    return entities

#function takes two arguments, a list and the item you want entities for in that list. For example, with this corpora, select a given
#item i for analysis--in this case i is just a document in the doc library already brought in
def get_entitylist(list, i):
    entitylist = []
    for entity in extract_entities(list[i]):
        #calls extract_entities on the given text
        item = '[' + entity.node + '] ' + ' '.join(c[0] for c in entity.leaves())
        #gets the entity node type and joins it with the leaf--basically the entity name in this case
        entitylist.append(item)
        #appends to the entitylist object
    return entitylist


# In[302]:

#uniq is a function to return only unique items for a given list
def uniq(input):
    output = []
    for x in input: 
        if x not in output:
            output.append(x)
    return output


# In[303]:

#prints each item in a list on it's own line
def printbyline(list):
    for item in list:
        print item


# In[161]:

#just a test of the entity extraction tool
entitysample = uniq(get_entitylist(corpora, 1))

printbyline(sorted(entitysample[:10]))


# In[304]:

#converts corpora from a list into a list of tuples where each tuple is the text of the document
i=0
corpora_processing = []
while i < len(corpora):
    corpora_processing.append(corpora[i:i+1])
    i += 1


# 
# 
# 
# 
# #The classification function
# Below, I specify the parameters of the classification function and use it on a training set: the first 50 documents in the corpus. I use the last 10 documents in the corpus as a test set.

# In[358]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

x_train = np.array(corpora[:50])
y_train = taglist[:50]

x_test = np.array(corpora[50:60])
y_test = taglist[50:60]

classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(3, 3), stop_words='english', min_df=0.1, max_df=0.75, max_features=500)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)

print predicted
len(predicted)


# In[359]:

for i in range(len(predicted)):
    print 'Predicted: ' + str(predicted[i])
    print 'Actual: ' +str(y_test[i])
    print


# In[338]:

sample_text = np.array(['In Sudan, there are many small businesses and we need to support them with NGOs and public-private partnerships.',
                        'In China, we have lots of new technology emerging. The USG should strategically plan around this',
                        'The embassy in Argentina is working hard on trade issues.'])
print classifier.predict(sample_text)


# In[ ]:



