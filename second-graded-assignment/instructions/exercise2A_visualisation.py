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
    
    sem_model.init_sims()
    return sem_model

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

def load_categories_words(words_filepath="wordnet_categorisation.txt"):
    '''Reads a tab-separated file with two columns and returns a dictionary 
    with elements in the first column as keys and elements in the second column as values'''
    my_taxonomy=dict()
    ### YOUR CODE HERE ###
    return my_taxonomy

def word_vec_list(sem_model, relevant_words):
    """ Build a list of word embeddings, vector_list, for the target words 
    given in relevant_words, by means of (1a)-(1c). 
    Since it is possible that not all target words are found in the semantic 
    model, a new list of words, word_list, will be also be created which will 
    only contain those words which are in sem_model.
    """
    word_list = []
    vector_list = []
    ### YOUR CODE HERE ###

    # (1a) iterate over the list 
        # (1b) retrieve the embedding (vector / numpy array) from the semantic model
            # (1c) appending the vector to the list embeddings_list, ...
            # ... and the word to word_list

    # return the lists
    return vector_list, word_list

### ### ### ###  TASKS  ### ### ### ###

# tip: to understand how the function plot_with_tsne works, check the script
# understanding_plot_with_tsne.py

# Load the semantic model
sem_model = load_sem_model()

# Load the taxonomy you created by searching for the target words' categories in WordNet (see exercise 1).
# You will need to fill in the function load_categories_words().

taxonomy = load_categories_words( "wordnet_categorisation.txt")
print("Taxonomy loaded:\n", taxonomy)

## Step 1
# Create one list of tokens (words) which are to be plotted with t-sne, tokens_to_plot,
# which contains both the words and the categories
tokens_to_plot=[]]

## Step 2
# Retrieve the embeddings (vectors) for the target words in tokens_to_plot from 
# the semantic model, vector_list using the function word_vec_list.
# This function also returns the list of tokens, token_labels, which have been 
# found in the semantic model (and for which teh vector has been retrieved). 
vector_list, token_labels = word_vec_list(sem_model, tokens_to_plot)

## Step 3
# Plot the words and categories with plot_with_tsne
#plot_with_tsne(vector_list, token_labels, outfile_name="vis_cats_words")


## Alternative solution to Steps 1-3 above:
# with different colours for words and categories
#  (i) start with 3 empty lists
vector_list = []
tokens_to_plot = []
token_colours = []

# fill in 'token_colours' with different colors for target words and categories
# hint:

# word_colours = ['blue']*len(word_labels)

## Plot the words and categories with plot_with_tsne, using different colours for the former

#plot_with_tsne(vector_list, tokens_to_plot, color_coding=token_colours, outfile_name="vis_cats_words_coloured")
