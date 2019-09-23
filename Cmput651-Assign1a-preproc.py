# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:10:33 2019

@author: zehra
"""

import os
import numpy as np
import pickle

basepath = os.path.join('C:\\','Zehra','fall2019','Cmput 651 Deep learning for NLP',\
                        'Assignments','Assignment1','aclImdb')

#Function to read input files into a python list
def readIMDBFiles(base_path,folder='train',subfolder='pos'):
    folderpath = os.path.join(base_path,folder,subfolder)
    
    file_list=[]
    for filename in os.listdir( folderpath ):
        filepath = os.path.join(folderpath,filename)    
        with open(filepath,'r',encoding='utf8') as file:
            contents = file.read()
            contents = contents.strip().lower() #strip whitespace, convert to lowercase
            file_list.append(contents)
    
    return file_list

#Read in training and test sets (pos is positive samples, neg is negative samples)
train_pos = readIMDBFiles(basepath,'train','pos')
train_neg = readIMDBFiles(basepath,'train','neg')
test_pos = readIMDBFiles(basepath,'test','pos')
test_neg = readIMDBFiles(basepath,'test','neg')

#Randomly shuffle the training set (both pos and neg samples), to ensure i.i.d.
np.random.shuffle(train_pos)
np.random.shuffle(train_neg)

#Choose the first 2500 samples (from both pos and neg) as the validation set (total 5,000 samples in val set)
#Keep the remaining samples in training set (total 20,000 samples in train set)
val_pos = train_pos[0:2500]
train_pos = train_pos[2500:]
val_neg = train_neg[0:2500]
train_neg = train_neg[2500:]


#Create X_train by putting together both pos and neg samples
#Also create y_train by assigning label '1' to pos samples, label '0' to neg samples
X_train = train_pos + train_neg
y_train = [1]*len(train_pos) + [0]*len(train_neg)

#Shuffle the entire training dataset
#Setting the same random seed ensures the correspondence between X and y is preserved
np.random.seed(3)
np.random.shuffle(X_train)
np.random.seed(3)
np.random.shuffle(y_train)

#Similar steps for validation set
X_val = val_pos + val_neg
y_val = [1]*len(val_pos) + [0]*len(val_neg)

np.random.seed(123)
np.random.shuffle(X_val)
np.random.seed(123)
np.random.shuffle(y_val)

#And similar again for test set
X_test = test_pos + test_neg
y_test = [1]*len(test_pos) + [0]*len(test_neg)

np.random.seed(321)
np.random.shuffle(X_test)
np.random.seed(321)
np.random.shuffle(y_test)


#Building the vocab based on words seen in X_train
vocab = dict()
for review in X_train:
   for w in review.split():
       if w in vocab:
           vocab[w] += 1
       else:
           vocab[w] = 1

#Sorting the vocab
vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

vocab = vocab[0:2000] #Choosing the 2000 most common words in vocab

#Constructing helper objects: id2word (list), word2id (dict)
id2word = [ wordtup[0] for wordtup in vocab ]
word2id = dict()
for i,word in enumerate(vocab):
    word2id[ word[0] ] = i


#Function to create feature vector given a review (using word2id dict)
#Returned vector has length same as size of vocab (2000 in this case).
#If a word in word2id dict appears in this review, the corresponding position in feature vec is set to 1.
def getReviewFeatureVec (review, word_dict=word2id):
    reviewFeatVec=np.zeros((1,len(word_dict)))
    for word in review.split():
        if word in word_dict: #set associated word_id in reviewFeatVec
            w_id = word_dict[word]
            reviewFeatVec[:,w_id] = 1
    return reviewFeatVec

#Construct a numpy array of size N_data x N_feature to hold all the samples
train_data = np.zeros((len(X_train),len(word2id))) #train_data: 20,000 x 2,000
for i,text in enumerate(X_train):
    train_data[i,:] = getReviewFeatureVec(text)

val_data = np.zeros((len(X_val),len(word2id))) #val_data: 5,000 x 2,000
for i,text in enumerate(X_val):
    val_data[i,:] = getReviewFeatureVec(text)
    
test_data = np.zeros((len(X_test),len(word2id))) #test_data: 25,000 x 2,000
for i,text in enumerate(X_test):
    test_data[i,:] = getReviewFeatureVec(text)

#Now pickle all the objects created, so we can load them easily later
pickle.dump((train_data, y_train, val_data, y_val, test_data, y_test), open("data.pkl", "wb"))
pickle.dump( (id2word, word2id), open("dicts.pkl", "wb") )






