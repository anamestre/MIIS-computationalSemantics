# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:08:08 2020

@author: Ana
"""

import os
import sys

import gensim
from gensim import matutils 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from numpy import dot
from numpy.linalg import norm

# Get list of hyponyms from "file".
def get_hyponyms(file):
    hypos, entities, concepts = [], [], []
    with open(file,  encoding='utf-8') as f:
        for line in f.readlines():
            hyponym, type_ = line.strip().split('\t')
            if hyponym == "riptide":
                print("--------------", hyponym.lower())
            hypos.append(hyponym)
        
            if type_ == "Entity":
                entities.append(hyponym)
            else:
                concepts.append(hyponym)
    return hypos, entities, concepts


# Returns a list of lists of hypernyms. Every sublist corresponds to a line on "file"
def get_hypernyms(file):
    categories = []
    with open(file) as f:
        for line in f.readlines():
            hypernyms = []
            for word in line.strip().split('\t'):
                if word not in categories:
                  hypernyms.append(word)
            categories.append(hypernyms)
    return categories


# Returs a dictionary of: Key:hyponym, Value:hypernyms from two files
def get_hypos_hypers(file_hypos, file_hypers):
    keys, _, _ = get_hyponyms(file_hypos)
    values = get_hypernyms(file_hypers)
    hypos_hypers = dict(zip(keys, values))
    return hypos_hypers


# From "hypo_vectors", obtains a list of sorted hyponyms by their similarity to "hypo"
def get_most_similar(sem_model, hypo, k, hypo_vectors):
  similarities = {}
  if ' ' in hypo:
      word1, word2 = hypo.split(" ", 1)
      h2 = word1 + "_" + word2
  elif '-' in hypo:
      word1, word2 = hypo.split("-", 1)
      h2 = word1 + "_" + word2
  else:
      h2 = hypo
  vect = retrieve_vector(sem_model, h2) 
  if vect is None:
    return similarities
  
  for hypo_ in hypo_vectors:
    vect2 = hypo_vectors[hypo_]
    if vect2 is not None:
      sim = cosine_similarity(vect, vect2)
      similarities[hypo_] =  sim
  #sorted_sim = sorted(similarities, key = lambda key: similarities[key], reverse = True)
  sorted_sim = sorted(similarities.items(), key = lambda x:x[1], reverse = True)
  return sorted_sim[:k]


# Retrieves a string of sorted hypernyms for "hyponym"
def get_hypernyms_by_hypo(sem_model, k, hyponym, hypo_hyper, hypo_vectors):
  if hyponym in sem_model:
    most_similar = get_most_similar(sem_model, hyponym, k, hypo_vectors)
    
    #print("______________________________")
    #print("Hyponym:", hyponym)
    #print("Most similar:", most_similar)
    top_hypernyms = {}
    similarity = {}
    for (hypo, sim) in most_similar:
          hypernyms = hypo_hyper[hypo] # Getting the list of hypernyms of every hypo.
          """print("Hypo:", hypo)
          print("Hypernym:", hypernyms)"""
          for hyper in hypernyms:
              #if hyper in sem_model:
              #  sim = sem_model.similarity(hyponym, hyper)
              if hyper in top_hypernyms.keys():
                  prev_value = top_hypernyms[hyper]
                  if sim > prev_value:
                      top_hypernyms[hyper] = sim
                      similarity[hyper] = sim
              else:
                  top_hypernyms[hyper] = sim
                  similarity[hyper] = sim
              #print("Sim:", top_hypernyms[hyper])
    
    res = sorted(top_hypernyms, key = lambda key: top_hypernyms[key], reverse = True)
    return res[:15], similarity
  else:
    return []  , {}


# Computes cosine similarity
def cosine_similarity(A, B):
    dot_val = dot(A, B)
    lens_val = norm(A) * norm(B)
    return dot_val / lens_val


# Retrieves vector for "word" in the "sem_model" semantic model.
def retrieve_vector(sem_model, word):
    # word is a multi-word-expression -- for simplicity, we assume it's a two-word expression
    if "_" in word:
        word1, word2 = word.split("_", 1)
        vector, _ = compose_words(sem_model, word1, word2)
        return vector
    
    if word not in sem_model:
        #print(word, "not found in the semantic model.")
        return None
    return sem_model[word.lower()]


# Retrives vector for phrases (compound words) "word1", "word2" in "sem_model"
def compose_words(sem_model, word1, word2):
    """
    Returns a synthetic vector built by computing 
    the component-wise sum of the embeddings of word1 and word2 and averaging the sum, 
    then normalising this mean vector to unit length (i.e., length == 1, i.e., the dot product is 1: [python command: np.dot(synthetic_vec, synthetic_vec)]).
    """
    if word1 in sem_model and word2 in sem_model:
        vec1 = retrieve_vector(sem_model, word1.lower())
        vec2 = retrieve_vector(sem_model, word2.lower())
        vecs = [vec1, vec2]
        synthetic_vec = matutils.unitvec(np.array(vecs).mean(axis=0)).astype(np.float32)
        #print(type(synthetic_vec))
        return synthetic_vec, word1 + "_" + word2
    """else:
        print(word1, "or", word2, "not found in the semantic model.")"""

    return None, None


def dict_of_vectors(hs, sem_model):
    res = {}
    for h in hs:
        if ' ' in h:
            word1, word2 = h.split(" ", 1)
            h2 = word1 + "_" + word2
        elif '-' in h:
            word1, word2 = h.split("-", 1)
            h2 = word1 + "_" + word2
        else:
            h2 = h
        res[h] = retrieve_vector(sem_model, h2)
    return res














spanish = True

if spanish:
    new_model = Word2Vec.load("spanish_word2vec.model")
    train_hypos = "SemEval2018-Task9/training/data/1C.spanish.training.data.txt"
    train_hypers = "SemEval2018-Task9/training/gold/1C.spanish.training.gold.txt"
    test_hypos = "SemEval2018-Task9/test/data/1C.spanish.test.data.txt"
    test_hypers = "SemEval2018-Task9/test/gold/1C.spanish.test.gold.txt"
    output_file = "SemEval2018-Task9/output_spanish.txt"
else:
    new_model = Word2Vec.load("english_word2vec.model")
    train_hypos = "SemEval2018-Task9/training/data/1A.english.training.data.txt"
    train_hypers = "SemEval2018-Task9/training/gold/1A.english.training.gold.txt"
    test_hypos = "SemEval2018-Task9/test/data/1A.english.test.data.txt"
    test_hypers = "SemEval2018-Task9/test/gold/1A.english.test.gold.txt"
    output_file = "SemEval2018-Task9/output.txt"

hypos_hypers_train = get_hypos_hypers(train_hypos, train_hypers)
hypos_train = hypos_hypers_train.keys()

hypos_hypers_test = get_hypos_hypers(test_hypos, test_hypers)
hypos_test = hypos_hypers_test.keys()

hypos_train_vector = dict_of_vectors(hypos_train, new_model)
hypos_test_vector = dict_of_vectors(hypos_test, new_model)

K = 10

results = {}
i = 1
for hypo in hypos_test:
    all_hypers, simi = get_hypernyms_by_hypo(new_model, K, hypo, hypos_hypers_train, hypos_train_vector)
    results[hypo] = all_hypers
    """print("---------------", i, hypo)
    for hyper in all_hypers:
        print(hyper, simi[hyper])
    i += 1"""
    
with open(output_file, 'w') as f:
    for hypo in results:
        if results[hypo] == []:
            f.write('\n')
        else:
            f.write('\t'.join(results[hypo]) + '\n')