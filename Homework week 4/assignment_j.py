# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:06:46 2019

@author: Carina Silberer
"""
import sys

import os

import numpy as np
# Needed for computing the dot product between two vectors
from numpy import dot

import gensim
# Needed for computing the unit length of a vector
from gensim import matutils 

# Needed for tsne & plotting in 2-dimensional space
import tsne
import matplotlib.pyplot as plt # for plotting
    
import nltk
from nltk.data import find


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


def plot_with_tsne(vectors, words, color_coding=None, outfile_name="tsne_solution"):
    # is vectors in the right data structure (numpy array)?
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # Apply tsne to project the word embeddings into a 2-dimensional space
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

''' Create a tsne graph for the vectors in data/word_phrase_embeddings.txt

Basically, it's reusing the code from inclass_word2vec_tsne.py with
another file for the semantic space (see function load_toy_semantic_model).

... and analyze it:

what kinds of expressions are rightly close together or far apart?
what kinds of similar expressions appear far apart? 
what kinds of dissimilar expressions appear close?

'''

word_list = []
with open('data/word_pairs_pruned.txt','r') as f:
    for line in f:
        for word in line.split():
           print(word) 
           word_list.append(word)
               
           
word_list2 = []
with open('data/single_words_pruned.txt','r') as g:
    for line in g:
        for word in line.split():
           print(word) 
           word_list2.append(word)
           

words_list = word_list+word_list2

print(words_list)

        
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


sem_model = load_sem_model()


listX=[]
vector_list=[] 

for word in words_list:
    if word in sem_model:
        print("The word  -  ", word , " -  *** Exists ***    in the semantic model" )
        vector_list.append(sem_model[word])
        listX.append(word)
    else:
        print("X")
        
        

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
    
from matplotlib.pyplot import figure
figure(num=None, figsize=(24, 10), dpi=80, facecolor='w', edgecolor='k')
plot_with_tsne(vector_list, listX)