##############################################
###############  IMPORTS    ##################
##############################################

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

##############################################
##############Data Preprocessing##############
##############################################

#Data Cleaning
import argparse
from nltk.stem.isri import ISRIStemmer
import sys

#Tokenization


##############################################
##############  MODEL TRAINING  ##############
##############################################

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


########################################################################################


##############################################
##############  PreProcessing  ##############
##############################################
#Import the data
df = pd.read_csv('all_poems.csv')
df.poem_text=df.poem_text.astype(str)

#Clean the data
df.drop(["poem_style"], axis=1, inplace=True)
df.drop(["poem_link"], axis=1, inplace=True)
df.drop(["poet_link"], axis=1, inplace=True)
df.drop(["poet_id"], axis=1, inplace=True)
df.drop(["poet_name"], axis=1, inplace=True)
df.drop(["poem_title"], axis=1, inplace=True)
df.drop(["poem_id"], axis=1, inplace=True)

#Downsampling
df.drop((df[df["poet_cat"]=="العصر العباسي"]).index[:13000], inplace=True)
df.drop((df[df["poet_cat"]=="العراق"]).index[:2000], inplace=True)
df.drop((df[df["poet_cat"]=="سوريا"]).index[:3000], inplace=True)
df.drop((df[df["poet_cat"]=="لبنان"]).index[:2000], inplace=True)

#Grouping categories 
def combine(x):
    if (( x == "العراق") or (x == "السعودية") or (x == "الإمارات") or (x == "البحرين") or (x == "الكويت") or (x == "قطر") or (x == "عمان") or  (x == "اليمن") ) :
        return "Arabian Peninsula"
    elif ( x == "فلسطين" or x == "سوريا" or x == "لبنان" or x == "الأردن") :
        return "Bilad al-Sham"
    elif ( x == "الجزائر" or x=="السودان" or x=="تونس" or x=="ليبيا" or x=="المغرب" or x=="مصر"):
        return "North African Countries"
    elif (x=="العصر الأندلسي"):
        return "Andalusia"
    elif (x == "العصر العباسي"):
        return "Abbasid Empire"
    else:
        return "Other"

df["category"] = df['poet_cat'].apply(combine) # create new column with specific values to conditions


df.drop(df[df["category"]=="Other"].index, axis=0, inplace=True) # Drop all other categories

#Preapring Data - Numpy Array

#Creating new dataframe for training 
new_data = df.drop(["poet_cat"], axis=1)
new_data = new_data.drop(["category"], axis=1)
dflabel = df[['category']]
new_data.insert(len(new_data.columns),"category",dflabel)

#########################################################################
############################  NUMPY ARRAYS  #############################
#########################################################################

# convert the array into a numpy array
arr = new_data.to_numpy()
# separate X and Y
Y = arr[:,1]
Y=Y.astype(str)
X = arr[:,0]
X=X.astype(str)
print(X)
print(Y)


#########################################################################
############################  Cleaning Data  #############################
#########################################################################


# Arabic light stemming for Arabic text
# takes a word list and perform light stemming for each Arabic words
def light_stem(text):
    words = text.split()
    result = list()
    stemmer = ISRIStemmer()
    for word in words:
        word = stemmer.norm(word, num=1)      # remove diacritics which representing Arabic short vowels
        if not word in stemmer.stop_words:    # exclude stop words from being processed
            word = stemmer.pre32(word)        # remove length three and length two prefixes in this order
            word = stemmer.suf32(word)        # remove length three and length two suffixes in this order
            word = stemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
            word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
#             word=stemmer.pro_w4(word)         #process length four patterns and extract length three roots
#             word=stemmer.pro_w53(word)        #process length five patterns and extract length three roots
#             word=stemmer.pro_w54(word)        #process length five patterns and extract length four roots
#             word=stemmer.end_w5(word)         #ending step (word of length five)
#             word=stemmer.pro_w6(word)         #process length six patterns and extract length three roots
#             word=stemmer.pro_w64(word)        #process length six patterns and extract length four roots
#             word=stemmer.end_w6(word)         #ending step (word of length six)
#             word=stemmer.suf1(word)           #normalize short sufix
#             word=stemmer.pre1(word)           #normalize short prefix
            
        result.append(word)
    return ' '.join(result)

cleaned_X=[]
for i in range (1):
    cleaned_X.append(light_stem(X[i]))
    print(cleaned_X)
    print("Nromal")
    print(X[i])


#########################################################################
############################  Cleaning Data  #############################
#########################################################################
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(X)
vocab_size = len(t.word_index) + 1


# stemmed=light_stem(t)


# integer encode the documents
encoded_docs = t.texts_to_sequences(X)
print(encoded_docs)


# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
data=load_vectors('wiki.ar.vec')


print('Loaded %s word vectors.' % len(embeddings_index))


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 301))
print(embedding_matrix)
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# define model
model = Sequential()
e = Embedding(vocab_size, 301, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
docs=np.asarray(padded_docs)
print("ho")
print(docs)
print(padded_docs)
print(labels)
labelss=np.asarray(labels)
model.fit(docs, Y, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))



