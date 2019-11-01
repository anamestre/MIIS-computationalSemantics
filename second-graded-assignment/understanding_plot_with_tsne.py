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

    plt.savefig(outfile_name+".png", format='png', dpi=200)
    plt.show()


# try this to understand how plot_with_tnse works

vector_list=[[1,2],[2,2],[3,3]]
tokens_to_plot=['one','two','three']
token_colours=['red','red','blue']
plot_with_tsne(vector_list, tokens_to_plot, color_coding=token_colours, outfile_name="test")
