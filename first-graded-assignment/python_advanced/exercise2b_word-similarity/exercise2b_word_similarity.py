# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:46:12 2019

@author: Carina Silberer
"""
import os
import sys

import gensim
#from gensim.models.word2vec import Word2Vec
import nltk
from nltk.data import find
import pandas
import numpy
from scipy.stats import spearmanr

def load_sim_benchmark(fname="SimLex-999/SimLex-999.txt"):
    graded_similarity = {}
    with open(fname) as f:
    # the first line contains the column names, we'll skip it
        f.readline()
        # iterates through each line in the reference file (see exercise1)
        # word1	word2	POS	SimLex999	conc(w1)	conc(w2)	concQ	Assoc(USF)	SimAssoc333	SD(SimLex)
        for line in f.readlines():
            word1, word2, _, simLex999, _, _, _, _, _, _ = line.strip().split("\t")
            graded_similarity[(word1, word2)] = simLex999
        return graded_similarity

def load_wordpairs(fname="SimLex-999/SimLex-999.txt"):
    with open(fname) as f:
    # the first line contains the column names, we'll skip it
        f.readline()
        # iterates through each line in the reference file (see exercise1)
        # word1	word2	POS	SimLex999	conc(w1)	conc(w2)	concQ	Assoc(USF)	SimAssoc333	SD(SimLex)
        word_pairs = []
        for line in f.readlines():
            word1, word2, _, _, _, _, _, _, _, _ = line.strip().split("\t")
            word_pairs.append((word1, word2))
        return word_pairs
    
    pass

def load_model(word2vec_modelfname=None, is_binary=False):
    if word2vec_modelfname is None:
        try:
            word2vec_modelfname = str(find('models/word2vec_sample/pruned.word2vec.txt'))
            is_binary = False
        except LookupError:
            nltk.download('word2vec_sample')
            word2vec_modelfname = str(find('models/word2vec_sample/pruned.word2vec.txt'))
            
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=is_binary)
    print("Semantic model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (len(model.vocab), model.vector_size))
    return model

def get_similarity_predictions(sem_model, word_pairs):
    # Iterate over list of word pairs and estimate their similarity using the semantic model
    similarity = {}
    for w1, w2 in word_pairs:
        similarity[(w1, w2)] = sem_model.similarity(w1, w2)
    return similarity

def evaluate(measure="Spearman", predictions=None, gold=None):
    return spearmanr(predictions, gold, nan_policy='raise')

def print_similarity_predictions(word_pairs, model_predictions, gold_ratings, outfname="similarity_predictions_word-pairs.txt"):
    with open(outfname, 'w') as file:
        file.write("Word1 \t Word2 \t Model prediction \t Gold rating")
        for w1, w2 in word_pairs:
            file.write('\n')
            file.write(w1 + "\t" + w2 + "\t" + str(model_predictions[(w1,w2)]) + "\t" + str(gold_ratings[(w1, w2)]))
    

if __name__=="__main__":
    benchmark_data_fname = "SimLex-999/SimLex-999.txt"

    # Load the semantic model containing word meaning representations
    # NOTE: replace "True" by "False" for loading the large model built on google news
    sample_model = False
        
    if sample_model:
        sem_model = load_model()
    else:
        word2vec_modelfname = 'C:/Users/Ana/Documents/Master/MIIS-computationalSemantics/Assignment1/week1-dtic/week1-dtic/GoogleNews-vectors-negative300.bin.gz'
        sem_model = load_model(word2vec_modelfname, is_binary=True)    
    
    # TODO: Load the word similarity benchmark
    graded_sim = load_sim_benchmark()
    
    # TODO: Estimate the similarity of all word pairs with the model 
    # ==> 
    word_pairs = load_wordpairs()
    predicted_sim = get_similarity_predictions(sem_model, word_pairs)
    # print(sim)
    
    # TODO: Print the model predictions and the human judgements / write them into a file
    print_similarity_predictions(word_pairs, predicted_sim, graded_sim)
    
    # TODO: Evaluate the model against the gold ratings using by correlating the scores
    # using Spearman's rho
    # ==> evaluate()
    predicts = list(predicted_sim.values())
    golds = list(graded_sim.values())
    print(evaluate("Spearman", predicts, golds))
    
    # TODO: See 3. Questions in the ex2b_instructions_word-similarity.txt
    
