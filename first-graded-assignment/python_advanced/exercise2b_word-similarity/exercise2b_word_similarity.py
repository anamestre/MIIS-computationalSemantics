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
    pass

def load_wordpairs(fname="SimLex-999/SimLex-999.txt"):
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
    pass

def evaluate(measure="Spearman", predictions=None, gold=None):
    pass

def print_similarity_predictions(word_pairs, model_predictions, gold_ratings, outfname="similarity_predictions_word-pairs.txt"):
    pass
    

if __name__=="__main__":
    benchmark_data_fname = "SimLex-999/SimLex-999.txt"

    # Load the semantic model containing word meaning representations
    # NOTE: replace "True" by "False" for loading the large model built on google news
    sample_model = False
        
    if sample_model:
        sem_model = load_model()
    else:
        word2vec_modelfname = 'GoogleNews-vectors-negative300.bin.gz'
        sem_model = load_model(word2vec_modelfname, is_binary=True)    
    
    # TODO: Load the word similarity benchmark
    # ==> load_sim_benchmark()
    
    # TODO: Estimate the similarity of all word pairs with the model 
    # ==> get_similarity_predictions()
    
    # TODO: Print the model predictions and the human judgements / write them into a file
    # ==> print_similarity_predictions()
    
    # TODO: Evaluate the model against the gold ratings using by correlating the scores
    # using Spearman's rho
    # ==> evaluate()
    
    # TODO: See 3. Questions in the ex2b_instructions_word-similarity.txt
    
