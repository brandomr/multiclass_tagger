
# coding: utf-8

# In[214]:

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs


# In[215]:

metadata = pd.read_csv("DocumentTagResults.csv")


# In[216]:

metadata.columns


# In[217]:

metadata['FileName'] = metadata['FileName'].replace(to_replace='.docx', value='.htm', regex=True)


# In[218]:

filenames = os.listdir('/Users/brandomr/Sites/docs')
#gets list of filenames in the document directory

len(filenames)
#tests how many files are in the docs folder


# In[219]:

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


# In[283]:

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


# In[275]:

#uniq is a function to return only unique items for a given list
def uniq(input):
    output = []
    for x in input: 
        if x not in output:
            output.append(x)
    return output


# In[286]:

#prints each item in a list on it's own line
def printbyline(list):
    for item in list:
        print item


# In[287]:

entitysample = uniq(get_entitylist(corpora, 1))


# In[278]:

printbyline(sorted(entitysample))


# In[288]:

sampletext = ['Brandon is the bomb diggety fresh. He lives in Washington DC in a small apartment with Barb. He loves her very much and likes going to work at the State Department. One day he will be rich and famous and have houses in New York City and San Francisco and Hong Kong and Los Angeles and San Diego.']


# In[289]:

get_entitylist(sampletext,0)


# In[ ]:



