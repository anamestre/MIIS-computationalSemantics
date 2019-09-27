# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:06:46 2019

@author: Carina Silberer
"""
import sys

# Possible error: "ModuleNotFoundError: No module named 'gensim'"
# ==> install gensim, e.g.:
# (in Anaconda Prompt, type the command)    pip install gensim 
# Possible error while installing gensim: "Permission denied"
# ==> run Anaconda Prompt "as administrator", try the above again
# OR install gensim using the Anaconda Navigator

try:
    import gensim
except ImportError:
    sys.stderr.write("Module gensim not installed. Please see the script exercise1_advanced.py for how to install it.")
    sys.exit()
    
import nltk
from nltk.data import find

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
    model.init_sims()
    return model

#### START SEMANTIC MODEL
# Load the semantic model containing word meaning representations
# NOTE: set sample_model to False for loading the large model built on google news
sample_model = True

if sample_model:
    sem_model = load_model()
else:
    word2vec_modelfname = 'GoogleNews-vectors-negative300.bin.gz'
    sem_model = load_model(word2vec_modelfname, is_binary=True)


# Python statements to use:

# Example word: "chicken"

# get word vector of a word:
sem_model["chicken"]

# compute the similarity between two words
sem_model.similarity('chicken','food')
sem_model.similarity('chicken','animal')

# retrieve the top n nearest neighbours of a word, e.g., topn=10
sem_model.most_similar("chicken", topn=10)

# retrieve the top n nearest neighbours of two-word phrases, e.g., topn=10
sem_model.most_similar(["raw", "chicken"], topn=10)
sem_model.most_similar(["flying", "chicken"], topn=10)

# Which word does not belong to the list of words?
sem_model.doesnt_match('chicken hen dog cat beef cow'.split())

	
# the cosine similarity between the means of the words in each of the two sets
sem_model.n_similarity(['hairy', 'dog'], ['furry', 'canine'])



