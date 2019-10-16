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
