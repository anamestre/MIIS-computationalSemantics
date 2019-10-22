# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:05:12 2019

@author: Ana
"""
import os
import sys
import gensim
from gensim import matutils 
import nltk
from nltk.data import find
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt # for plotting
import numpy as np
import tsne

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
    # Is vectors in the right data structure (numpy array)?     
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # Apply t-sne to project the word embeddings into a 2-dimensional space
    Y = tsne.tsne(X=vectors, no_dims=2, initial_dims=int(len(words)/2), perplexity=5.0, max_iter=1000)

    # Let's plot the solution:
    if color_coding is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c=color_coding)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], c = "m")

    # Let's add the words to the plot:
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        if ' ' in label:
            color = "c"
        else:
            color = "m"
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=9, c = color)

    plt.savefig(outfile_name+".png", format='png')
    plt.show()


flatten = lambda l: [item for sublist in l for item in sublist]

def read_single_words(data):
    words_list = []
    with open(data,'r') as data:
        for line in data.readlines():
            words = line.strip().split(" ")
            words_list.append(words)
    print(words_list)
    return flatten(words_list)

def treat_single_words(words):
    sem_vectors = []
    words_vector = []
    for w in set(words):
        if w in sem_model:
            print("In model: " + w)
            sem = sem_model[w]
            sem_vectors.append(sem)
            words_vector.append(w)
        else:
            print("XXXXX Not in model: " + w)
    return sem_vectors, words_vector

def read_pairs(data):
    words_list = []
    with open(data,'r') as data:
        for line in data.readlines():
            words = line.strip().split(" ")
            words_list.append(words)
    #print(words_list)
    return words_list

def treat_pairs(pairs):
    sem_vectors = []
    words_vector = []
    for w1, w2 in pairs:
        if w1 in sem_model and w2 in sem_model:
            sem_vec = []
            sem_vec.append(sem_model[w1])
            sem_vec.append(sem_model[w2])
            
            sem = np.array(sem_vec)
            sem_average = sem.mean(axis=0) # compute average
            pair_vec = matutils.unitvec(sem_average).astype(np.float32) # normalize
            
            sem_vectors.append(pair_vec)
            words_vector.append(str(w1 + " " + w2))
            
    return sem_vectors, words_vector


sem_model = load_sem_model()

data_single = "data/single_words_pruned.txt"
single_list = read_single_words(data_single)
sem_vectors_s, words_vector_s = treat_single_words(single_list)

data_pairs = "data/word_pairs_pruned.txt"
pairs_list = read_pairs(data_pairs)
treat_pairs(pairs_list)
sem_vectors_p, words_vector_p = treat_pairs(pairs_list)

sem_vectors = sem_vectors_s + sem_vectors_p
words_vector = words_vector_s + words_vector_p

figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plot_with_tsne(sem_vectors, words_vector)