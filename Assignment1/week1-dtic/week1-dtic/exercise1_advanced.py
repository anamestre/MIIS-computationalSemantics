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
    sys.stderr.write("Module gensim not installed. Please see the script test.py for how to install it.")
    sys.exit()
    
import nltk
from nltk.data import find

if __name__=="__main__":  
    # Load the semantic model
    # Possible error: Lookup Error. See also the instructions in the terminal for downloading the data.
    # Load the semantic model containing word meaning representations
    # NOTE: set sample_model to False for loading the large model built on google news
    sample_model = False
    
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
    
    words = ["apple", "sweet", "allow"]
    '''  
    # a) 20 nearest neighbors of the words in a space built on the google news-gram corpus, using
    # the word2vec semantic model.
    for w in words:
        neighbors = sem_model.most_similar(w, topn = 20)
        print("For the word %d here are the 20 nearest neighbors and its similarity score: " % (w))
        for n in neighbors:
            print("  - Word: %d, Similarity score: %d" % (n[0], n[1]))
    
    # b) 10 nearest neighbors for two-word phrases that you associate to different senses of the 
    # word (e.g., for "mouse", "use mouse" vs. "furry mouse").
    two_words = [["big", "apple"], [] ]
    #print(sem_model.most_similar(two_words, topn = 10))
    
    # c) Cosine similarity between the word and:
    # - one (near-)synonym, ---> fruit, sugary, concede
    # - one topically-related word,  ---> pear, sour, acknowledge
    # - one hypernym,---> fruit, flavour, provide
    # - one antonym (if available)  ---> salty, forbid '''
    
    synonyms = ["fruit", "sugary", "concede"]
    topic_related = ["pear", "sour", "acknowledge"]
    hypernyms = ["fruit", "flavour", "provide"]
    antonyms = ["salty", "forbid"]
    
    res_synonyms = map(lambda x,y: sem_model.similarity(x, y), list(zip(words, synonyms)))
    print(*res_synonyms)    
    

        
