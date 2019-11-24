# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:22:27 2019

@author: Ana
"""
import nltk, gensim, sys
from gensim import matutils
from numpy import np
from nltk.data import find

# Get list of hyponyms from "file".
def get_hyponyms(file):
    hypos, entities, concepts = [], [], []
    with open(file) as f:
        for line in f.readlines():
            hyponym, type_ = line.strip().split('\t')
            hypos.append(hyponym)
        
            if type_ == "Entity":
                entities.append(hyponym)
            else:
                concepts.append(hyponym)
    return hypos, entities, concepts

# Get list of all possible hypernyms from "file".
def get_vocabulary(file):
    hyper = []
    with open(file, encoding="utf8") as f:
        for line in f.readlines():
            hyper.append(line)

    return hyper

# Get a dictionary such that dictionary[hyponym] = [list of gold hypernyms]
def get_gold(hypos, file):
    hypo_hyper = {}
    with open(file, encoding="utf8") as f:
        i = 0
        for line in f.readlines():
            hypernyms_list = list(line.strip().split('\t'))
            hypo_hyper[hypos[i]] = hypernyms_list
            i += 1

    return hypo_hyper

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
    if word1 in sem_model and word2 in sem_model:
        vec1 = retrieve_vector(sem_model, word1)
        vec2 = retrieve_vector(sem_model, word2)
        vecs = [vec1, vec2]
        synthetic_vec = matutils.unitvec(np.array(vecs).mean(axis=0)).astype(np.float32)
        #print(type(synthetic_vec))
        return synthetic_vec, word1 + "_" + word2
    else:
        print(word1, "or", word2, "not found in the semantic model.")

    return None, None

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

x, _, _ = get_hyponyms("SemEval2018-Task9/trial/data/1A.english.trial.data.txt")
y = get_vocabulary("SemEval2018-Task9/vocabulary/1A.english.vocabulary.txt")
w = get_gold(x,"SemEval2018-Task9/trial/gold/1A.english.trial.gold.txt")