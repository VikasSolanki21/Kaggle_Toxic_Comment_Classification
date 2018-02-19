import numpy as np # linear algebra
import pandas as pd
import re
import string
import gensim
import enchant
import enchant.checker
from enchant.checker.CmdLineChecker import CmdLineChecker
from collections import Counter
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = train.iloc[:,:2].append(test,ignore_index=True)
#train = train.sample(frac=1

wnl = WordNetLemmatizer()
def clean_text(text):
    
    text = " " + text + " "
    
    text= re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," ",str(text.lower())) #IP's
    text= re.sub("http://.*com"," ",str(text)) #Links
    text =re.sub("\\n"," ",text)
    text = re.sub("’", "'", text) # special single quote
    text = re.sub("`", "'", text) # special single quote
    text = re.sub("“", '"', text) # special double quote
    
    text = re.sub(r"what's", "what is", text)
    text = re.sub("\'s", " ", text)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    
    text= re.sub(r'\d{2}:\d{2}', ' ', text) #Removing timestamp
    text = text.replace('(utc)','')
    
    #Replacing numbers
    text = re.sub('[0-9]', '', text)
    
    #Removing iterations
    #text=''.join(ch[:2] for ch, _ in itertools.groupby(text))
    
    #Punctuations
    text = re.sub('[.]',' . ',text)
    text = re.sub(',',' , ',text)
    text = re.sub('[-]',' - ',text)
    text = re.sub('[:]',' : ',text)
    text = re.sub('[;]',' ; ',text)
    text = re.sub('[?]', ' ? ',text)
    text = re.sub('[!]',' ! ',text)
    text = re.sub('[&]',' & ',text)
    
    text = re.sub('["#%\'()*+/<=>@[\\]^_{|}~]',' ',text)
    
    #Removing stopwords
    #text = " ".join([i for i in text.split() if i not in stop])
   
    #Lemmatisation
    text = " ".join([wnl.lemmatize(x,pos='v') for x in text.split()])
    #text = " ".join([wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(text))])
      
    return text

df['filtered_text']=df['comment_text'].apply(lambda x:clean_text(x))
df_train = df.iloc[:len(train),:]
train = pd.concat([df_train,train],axis=1).iloc[:,2:]
test = df.iloc[len(train):,:].reset_index(drop=True)
train.to_csv('filtered_train.csv',index=False)
test.to_csv('filtered_test.csv',index=False)


