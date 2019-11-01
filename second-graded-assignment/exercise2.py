# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:23:21 2019

@author: Ana
"""

import os, sys, gensim, tsne, nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import matutils 
from nltk.data import find
from matplotlib.pyplot import figure

def load_sem_model(sample_model=True):
    if sample_model:
        try:
            word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        except LookupError:
            nltk.download('word2vec_sample')
        word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    else:
        word2vec_modelfname = 'GoogleNews-vectors-negative300.bin.gz'
        try:
            sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=True)
            print("Google-news model of %d words, each represented by %d-dimensional vectors,    successfully loaded." % (len(sem_model.vocab), sem_model.vector_size))
        except FileNotFoundError:
            sys.stderr.write("Model file with name %s not found in directory. Please download it from %s\n." % (word2vec_modelfname, "https://code.google.com/p/word2vec/"))
            
    vocab_size = len(sem_model.vocab)
    embedding_size = sem_model.vector_size
    print("Model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (vocab_size, embedding_size))
    # We want all vectors to have unit length
    #sem_model.init_sims()
    return sem_model


def plot_with_tsne(vectors, words, color_coding = None, outfile_name="tsne_solution"):
    # Is vectors in the right data structure (numpy array)?
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # Apply t-sne to project the word embeddings into a 2-dimensional space
    Y = tsne.tsne(X=vectors, no_dims=2, initial_dims=int(len(words)/2), perplexity=5.0, max_iter=1000)

    # Let's plot the solution:
    if color_coding is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c = color_coding)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], c = "black")

    # Let's add the words to the plot:
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=9)

    plt.savefig(outfile_name+".png", format='png', dpi=200)
    plt.show()


def load_categories_words(words_filepath="wordnet_categorisation.txt"):
    '''Reads a tab-separated file with two columns and returns a dictionary 
    with elements in the first column as keys and elements in the second column as values'''
    my_taxonomy = defaultdict(list)
    with open(words_filepath) as f:
        for line in f.readlines():
            word, category = line.split()
            my_taxonomy[word].append(category)
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
    for word in relevant_words:
        sem = retrieve_vector(sem_model, word)
        if sem is not None:
            vector_list.append(sem)
            word_list.append(word)
    return vector_list, word_list


def retrieve_vector(sem_model, word):
    # word is a multi-word-expression -- for simplicity, we assume it's a two-word expression
    if "_" in word:
        word1, word2 = word.split("_", 1)
        vector, _ = compose_words(sem_model, word1, word2)
        return vector
    
    if word not in sem_model:
        print(word, "not found in the semantic model.")
        return None
    return sem_model[word]


def compose_words(sem_model, word1, word2):
    """
    Returns a synthetic vector built by computing 
    the component-wise sum of the embeddings of word1 and word2 and averaging the sum, 
    then normalising this mean vector to unit length (i.e., length == 1, i.e., the dot product is 1: [python command: np.dot(synthetic_vec, synthetic_vec)]).
    """
    print(word1, word2)
    if word1 in sem_model and word2 in sem_model:
        vec1 = retrieve_vector(sem_model, word1)
        vec2 = retrieve_vector(sem_model, word2)
        vecs = [vec1, vec2]
        synthetic_vec = matutils.unitvec(np.array(vecs).mean(axis=0)).astype(np.float32)
        return synthetic_vec, word1+"_"+word2
    else:
        print(word1, "or", word2, "not found in the semantic model.")

    return None, None


def average_vector(sem_model, word_list):
    """
    Returns a synthetic vector built by computing 
    the component-wise sum of the embeddings of the words in word_list and averaging the sum. 
    This mean vector is then normalised to unit length (i.e., length == 1, i.e., the dot product is 1: 
        [python command: np.dot(synthetic_vec, synthetic_vec)]).
    
    This function is a generalisation of compose_words, which expects two words, both contained in the semantic model, 
    to an arbitrary number of words, some of which may not be in the semantic space.
    """
    vectors = []
    words_found = []
    for w in word_list:
        if w in sem_model:
            words_found.append(w)
            sem = retrieve_vector(sem_model, w)
            vectors.append(sem)
    if (len(vectors) > 0):
        synthetic_vec = matutils.unitvec(np.array(vectors).mean(axis = 0)).astype(np.float32)
        return synthetic_vec, words_found

    return None, None


def invert_dict(dictionary):
    new_dictionary = defaultdict(list)
    for word in dictionary.keys():
        for cat in dictionary[word]:
            new_dictionary[cat].append(word)
    return new_dictionary

def read_input(file):
    with open(file) as f:
        input_list = [line.strip() for line in f.readlines()]
    return input_list


sem_model = load_sem_model()
taxonomy = load_categories_words("wordnet_categorisation.txt") # Dict[word] = [categories]
#print("Taxonomy loaded:\n", taxonomy)
## Steps A + B: 
# A: retrieve vectors for words and categories
# B: build prototype vectors for each category, from the respective categories' members
tokens_to_plot = []
token_colours = []
vectors_to_plot = []
# For B: build a dictionary from categories to a list of their respective members
category_to_words = invert_dict(taxonomy)
categories = read_input("categories_battig.txt") # Categories == hypernyms?¿
words = read_input("words_battig.txt")

vectors_to_plot, tokens_to_plot = word_vec_list(sem_model, words)
token_colours = ["blue"]*len(tokens_to_plot)
tokens_to_plot = ['']*len(tokens_to_plot)

vectors_to_plot_c, tokens_to_plot_c = word_vec_list(sem_model, categories)
token_colors_c = ["magenta"]*len(tokens_to_plot_c)

vectors_to_plot += vectors_to_plot_c
tokens_to_plot += tokens_to_plot_c
token_colours += token_colors_c
        
## This should plot words and categories:
# plot_with_tsne(vectors_to_plot, tokens_to_plot, color_coding=token_colours, outfile_name="vis_cats_words_coloured")

# Step B: build prototype vector for each category, using the categorisation dictionary category_to_words
#print("Computing prototype vectors from", category_to_words)
for (category, words) in category_to_words.items():
    average, tokens = average_vector(sem_model, words)
    if average is not None and tokens is not None:
        tokens_to_plot.append(category.upper())
        token_colours.append("yellow")
        vectors_to_plot.append(average)

# Steps A + B: Plot the words, categories, and prototypes with plot_with_tsne, using different colours for them
figure(num = None, figsize = (15, 15), dpi = 80, facecolor = 'w', edgecolor = 'k')
plot_with_tsne(vectors_to_plot, tokens_to_plot, color_coding = token_colours, outfile_name = "vis_cats_words_prototypes_coloured")