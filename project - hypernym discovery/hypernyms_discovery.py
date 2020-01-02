# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31

@author: Ana
"""

import os
import sys

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class AdaptedLineSentence(object):
    """
    Adapted source code from LineSentence (in gensim.models.word2vec).
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, source, lowercase=False):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.lowercase = lowercase is True
        self.lines_read = 0
        
    def preprocess(self, line):
        if self.lowercase is True:
            return line.lower()
        return line

    def __iter__(self):
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                yield self.preprocess(utils.to_unicode(line)).split()
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with gensim.utils.smart_open(self.source) as fin:
                for line in fin:
                    print ("something")
                    yield self.preprocess(gensim.utils.to_unicode(line)).split()
                    # uncomment this to first test that everything works, before you train the model o the full corpus
                    if self.lines_read > 10000000:
                        return
                    self.lines_read += 1
           
            
def get_model(corpus):
    my_reader = AdaptedLineSentence(corpus, lowercase=True)
    model = Word2Vec(my_reader, sg=1, size=300, workers=1) # train the model, adapt the parameters as you desire
    model.save("word2vec_lowercased.%s.model" % (corpus.split(".")[0]))  # save the model
    return model


# Get list of hyponyms from "file".
def get_hyponyms(file):
    hypos, entities, concepts = [], [], []
    with open(file,  encoding='utf-8') as f:
        for line in f.readlines():
            hyponym, type_ = line.strip().split('\t')
            hypos.append(hyponym)
        
            if type_ == "Entity":
                entities.append(hyponym)
            else:
                concepts.append(hyponym)
    return hypos, entities, concepts


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


def get_hypos_hypers(file_hypos, file_hypers):
    keys, _, _ = get_hyponyms(file_hypos)
    values = get_hypernyms(file_hypers)
    hypos_hypers = dict(zip(keys, values))
    return hypos_hypers


def get_hypernyms_by_hypo(sem_model, k, hyponym, hypo_hyper):
  k = 10
  if hyponym in sem_model:
    most_similar = sem_model.most_similar(hyponym, topn = k) # Get k most similar hyponyms
    print("______________________________")
    print("Hyponym:", hyponym)
    print("Most similar:", most_similar)
    top_hypernyms = {}
    for (hypo, sim) in most_similar:
        if hypo in hypo_hyper:
          hypernyms = hypo_hyper[hypo] # Getting the list of hypernyms of every hypo.
          print("Hypo:", hypo)
          print("Hypernym:", hypernyms)
          for hyper in hypernyms:
              if hyper in top_hypernyms.keys():
                  prev_value = top_hypernyms[hyper]
                  if sim > prev_value:
                      top_hypernyms[hyper] = sim
              else:
                  top_hypernyms[hyper] = sim
              print("Sim:", top_hypernyms[hyper])
    
    res = ' '.join(sorted(top_hypernyms, key = lambda key: top_hypernyms[key]))
    return res
  else:
    return None  




######################################################################
############################ Main script #############################
######################################################################


# hypos_hypers_train = dict key = hyponym, value = hypernym
train_hypos = "/content/drive/My Drive/Estudis/Master/1.1 Computational Semantics/data/1C.spanish.training.data.txt"
train_hypers = "/content/drive/My Drive/Estudis/Master/1.1 Computational Semantics/data/1C.spanish.training.gold.txt"
hypos_hypers_train = get_hypos_hypers(train_hypos, train_hypers)
print(hypos_hypers_train)
hypos_train = hypos_hypers_train.keys()

test_hypos = "/content/drive/My Drive/Estudis/Master/1.1 Computational Semantics/data/1C.spanish.test.data.txt"
test_hypers = "/content/drive/My Drive/Estudis/Master/1.1 Computational Semantics/data/1C.spanish.test.gold.txt"
hypos_hypers_test = get_hypos_hypers(test_hypos, test_hypers)
hypos_test = hypos_hypers_test.keys()

K = 10
sem_model = get_model("1A_en_UMBC_tokenized.tar.gz")

results = {}
print(hypos_test)
for hypo in hypos_test:
    all_hypers = get_hypernyms_by_hypo(new_model, K, hypo, hypos_hypers_train)
    results[hypo] = all_hypers