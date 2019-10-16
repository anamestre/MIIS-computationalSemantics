# -*- coding: utf-8 -*-
"""
Created 2019

@author: Carina Silberer
"""

import os
import sys

### ### ### ### THIRD-PATY PACKAGES ### ### ### ###
# We import third-party packages which are not directly in python distribution. That is, they may need to be installed by you.

import numpy as np

import gensim
# The following is needed for computing the unit length of a vector
from gensim import matutils 

# Needed for t-sne & plotting in 2-dimensional space; see tsne.py 
# in this folder if you are curious 
# t-sne (t-Distributed Stochastic Neighbor Embedding): https://lvdmaaten.github.io/tsne/
import tsne
import matplotlib.pyplot as plt # for plotting

# The Natural Language Toolkit: https://www.nltk.org/
# This toolkit is a standard for linguistic processing of text data -- you will probably work with it in the introductory course on computational linguistics in the second term. 
# Here, we just use it to load pruned word2vec GoogleNews embeddings (the "sample model").
import nltk
from nltk.data import find


### ### ### ### OUR CODE STARTS HERE ### ### ### ###

def load_sem_model(sample_model=True):
    # Load the semantic model
    # Possible error: Lookup Error. See also the instructions in the terminal for downloading the data.
    # Load the semantic model containing word meaning representations
    # NOTE: set sample_model to False for loading the large model built on google news
    if sample_model:
        try:
            word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        except LookupError:
            nltk.download('word2vec_sample')
        word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    else:
        # Load large Google-news semantic model
        word2vec_modelfname = 'GoogleNews-vectors-negative300.bin.gz'
        try:
            sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=True)
            print("Google-news model of %d words, each represented by %d-dimensional vectors,    successfully loaded." % (len(sem_model.vocab), sem_model.vector_size))
        except FileNotFoundError:
            sys.stderr.write("Model file with name %s not found in directory. Please download it from %s\n." % (word2vec_modelfname, "https://code.google.com/p/word2vec/"))
            
    vocab_size = len(sem_model.vocab)
    embedding_size = sem_model.vector_size
    print("Model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (vocab_size, embedding_size))
    
    return sem_model

def load_toy_semantic_model(filename = "data/short-cooccurrence-matrix.txt"):
    # the following command uses as is a command of gensim to load a semantic space
    toy_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)
    return toy_model


def plot_with_tsne(vectors, words, color_coding=None, outfile_name="tsne_solution"):
    # Is vectors in the right data structure (numpy array)?
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # Apply t-sne to project the word embeddings into a 2-dimensional space
    Y = tsne.tsne(X=vectors, no_dims=2, initial_dims=int(len(words)/2), perplexity=5.0, max_iter=1000)

    # Let's plot the solution:
    if color_coding is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c=color_coding)
    else:
        plt.scatter(Y[:, 0], Y[:, 1])

    # Let's add the words to the plot:
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=9)

    plt.savefig(outfile_name+".png", format='png')
    plt.show()


### ### ### ###  TASKS  ### ### ### ###

### ### ### ###
### PART 1  ###
### ### ### ###

# TODO:
# Call the following statements and make sure you understand what they do
# TIP for anaconda users: use F9 to run only the current line

sem_model = load_toy_semantic_model()

###
print("What does the statement below do?")
print(sem_model["hot"])

###
print("What do the statements below do?")
print("hot" in sem_model)
print("ahjr" in sem_model)

### ### ### ###
### PART 2  ###
### ### ### ###

### Now we are going to compose two vectors by averaging them:
### we are going to compose "hot" and "dog" into "hot dog"

# 1. Our vectors:
word1 = "hot"
word2 = "dog"
vec1 = sem_model[word1]
vec2 = sem_model[word2]
print(word1, vec1)
print(word2, vec2)

# 2. we put them into a list:
vectors = []
vectors.append(vec1)
vectors.append(vec2)
print("vectors: ", vectors)

# 3. we compute the average vector (cf. "prototype" in the Annual Review of Linguistics paper):
# 3a. We convert the list to an array in numpy. This is just so we can directly call 
# the 'mean' function below.
vectors_arr = np.array(vectors)
# 3b. We compute the mean of the vectors. 
# axis=0 means "do the average by column" (axis=1 would do it by row)
# that is, for each dimension, give as value the average of all the values of that dimension
average_vec = vectors_arr.mean(axis=0)
print("average_vec: ", average_vec)

# 4. then, we reduce the resulting vector to unit length (we normalize it)
# do not worry about this command, just use it as is:
hotdog_vec = matutils.unitvec(average_vec).astype(np.float32)
print("hotdog_vec: ", hotdog_vec)

# DONE :) 


### ### ### ###
### PART 3  ###
### ### ### ###


# finally, we are going to create 2-dimensional tsne graphs:

sem_model = load_sem_model()
 
turney_littman_positive=["good", "nice", "excellent", "positive", "fortunate", "correct", "superior"]
turney_littman_negative=["bad", "nasty", "poor", "negative", "unfortunate", "wrong", "inferior"]
turney_littman = turney_littman_positive + turney_littman_negative

###
## Task: For each word in turney_littman (hint: for-loop), check whether the word is in the semantic model. 
# If this is the case, 
# - retrieve the vector, and append it to the vector_list, and 
# - append the word to word_list

# List with words
word_list = []
# List with vectors (embeddings) corresponding to the words in word_list
vector_list = []
# TODO Put here your code to the task

# TODO: once you have succeeded creating the two lists, uncomment the following function 
# (see above for the code; but do not worry if you don't understand it),
# inspect the plot you will see (to see it better: open the file 'tsne_solution.png' in your current working directory) 
# and explain your observations to each other. 
####  plot_with_tsne(vector_list, word_list)

# TODO: run the tsne command a couple more times and observe what happens. 


### ### ### ### ### ### ### 
### OPTIONAL: CALCULUS  ###
### ### ### ### ### ### ### 

## Revise the cosine formula and check this:

# This is how to compute cosine similarity with gensim:
sem_model.similarity(word1, word2)

# and the dot product (nominator in cosine similarity formula), the function is already implemented by numpy (np), so we just use it:
np.dot(vec1, vec2)

# denominator in cosine similarity formula
vec1_unitlength = matutils.unitvec(vec1).astype(np.float32)
vec2_unitlength = matutils.unitvec(vec2).astype(np.float32)

# Compare
np.dot(vec1_unitlength, vec2_unitlength)

#to 
sem_model.similarity(word1, word2)

# make sure you understand why they give the same result
