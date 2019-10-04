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

"""
try:
    import gensim
except ImportError:
    sys.stderr.write("Module gensim not installed. Please see the script setup.py for how to install it.")
    sys.exit()

if __name__=="__main__":
    """
    Load the semantic model.
    Possible error: Lookup Error. See also the instructions in the terminal for downloading the data.
    Load the semantic model containing word meaning representations
    NOTE: set pruned_model to False for loading the large model built on google news,
    i.e., pruned_model = False
    """
    pruned_model = True

    if pruned_model is True:
        word2vec_modelfname = 'data/pruned.GoogleNews-vectors-negative300.txt'
        sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=False)
    else:
        # Load large Google-news semantic model
        word2vec_modelfname = 'data/GoogleNews-vectors-negative300.bin.gz'
        try:
            sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=True)
            print("Google-news model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (
                    len(sem_model.vocab), sem_model.vector_size))
        except FileNotFoundError:
            sys.stderr.write("Model file with name %s not found in directory. Please download it from %s\n(direct link to google drive: %s)." % (word2vec_modelfname, "https://code.google.com/p/word2vec/", "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing"))

    vocab_size = len(sem_model.vocab)
    embedding_size = sem_model.vector_size
    print("Model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (
            vocab_size, embedding_size))



''' try out the following commands in the console

# NOTE that if pruned_model = True, you are working with a small model (only 583 words)
# try it at home with the full model, if you can

# get word vector of a word:
sem_model["achieve"] # this will just be a big list of numbers...

# compute the similarity between two words
sem_model.similarity('achieve','accomplish')
sem_model.similarity('achieve','try')

# retrieve the top n nearest neighbours of a word, e.g., topn=10
sem_model.most_similar("achieve", topn=10)

# retrieve the top n nearest neighbours of two-word phrases, e.g., topn=10
sem_model.most_similar(["great", "summer"], topn=10)

# Which word does not belong to the list of words?
sem_model.doesnt_match('kitchen bathroom bedroom television'.split())

# the cosine similarity between the means of the words in each of the two sets
sem_model.n_similarity(['great', 'summer'], ['bad', 'winter'])
'''
