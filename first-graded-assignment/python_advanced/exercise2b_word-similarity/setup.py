# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:06:46 2019

@author: Carina Silberer
"""
import sys

"""
Possible errors:
================

"ModuleNotFoundError: No module named 'gensim'"
==> install gensim:
    IF YOU USE ANACONDA, use one of the following:
        A. in Anaconda Prompt, type the command: conda install -c anaconda gensim
            Possible error in Windows while installing gensim: "Permission denied"
                ==> run Anaconda Prompt "as administrator", try the above again
        B. install gensim using the Anaconda Navigator (as administrator)

    ELSE, you will know how to install packages :)

See also: https://radimrehurek.com/gensim/install.html

"ModuleNotFoundError: No module named 'nltk'"
==> install nltk (analogous as above)
See also: https://www.nltk.org/install.html
"""
try:
    import gensim
except ImportError:
    sys.stderr.write("\nModule gensim not installed. Please see the script setup.py for how to install it.")
    sys.exit()

try:
    import nltk
    from nltk.data import find
except ImportError:
    sys.stderr.write("\nModule nltk not installed. Please see the script setup.py for how to install it.")
    sys.exit()

# standard libraries used for scientific or data mining purposes
import pandas
import numpy
from scipy.stats import spearmanr

def load_model(word2vec_modelfname=None, is_binary=False):
    if word2vec_modelfname == 'GoogleNews-vectors-negative300.bin.gz':
        try:
            sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=True)
            print("Google-news model of %d words, each represented by %d-dimensional vectors,    successfully loaded." % (len(sem_model.vocab), sem_model.vector_size))
            return sem_model
        except FileNotFoundError:
            sys.stderr.write("Model file with name %s not found in directory. Please download it from %s\n(direct link to google drive: %s)." % (word2vec_modelfname, "https://code.google.com/p/word2vec/", "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing"))
            return None

    if word2vec_modelfname is None:
        try:
            word2vec_modelfname = str(find('models/word2vec_sample/pruned.word2vec.txt'))
            is_binary = False
        except LookupError:
            nltk.download('word2vec_sample')
            word2vec_modelfname = str(find('models/word2vec_sample/pruned.word2vec.txt'))

    sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=is_binary)
    print("Semantic model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (len(sem_model.vocab), sem_model.vector_size))
    return sem_model

if __name__=="__main__":
    sample_model = True

    # Load the benachmark data
    simlex = pandas.read_csv("SimLex-999/SimLex-999.txt", sep="\t")
    print("SimLex-999 benchmark successfully loaded. It contains %d word pairs." % (len(simlex)))

    if sample_model:
        sem_model = load_model()
    else:
        word2vec_modelfname = 'GoogleNews-vectors-negative300.bin.gz'
        sem_model = load_model(word2vec_modelfname, is_binary=True)
