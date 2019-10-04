# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:23:46 2019

@author: Carina Silberer
"""
import sys

try:
    import gensim
except ImportError:
    sys.stderr.write("Module gensim not installed. Please see the script setup.py for how to install it.")
    sys.exit()
 

def load_sem_model(pruned_model):
    """
    Loads the semantic model.
    Possible error: Lookup Error. See also the instructions in the terminal for downloading the data.
    Load the semantic model containing word meaning representations
    NOTE: set sample_model to False for loading the large model built on google news
    sample_model = False
    """
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
            sys.stderr.write("Model file with name %s not found in directory. Please download it from %s\n." % (
                    word2vec_modelfname, "https://code.google.com/p/word2vec/"))
            
    vocab_size = len(sem_model.vocab)
    embedding_size = sem_model.vector_size
    print("Model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (
            vocab_size, embedding_size))
    return sem_model

# set   pruned_model = False   if you want to load the full GoogleNews vectors
pruned_model = True
sem_model = load_sem_model(pruned_model)

column_separator = "\t" # the columns in the reference file are separated by tabs
tp = 0.0 # counter for the true positives
fp = 0.0 # counter for the false positives
tn = 0.0 # counter for the true negatives
fn = 0.0 # counter for the false negatives
scores = {"tp": {'A': 0.0, 'N': 0.0, 'V': 0.0},
          "fp": {'A': 0.0, 'N': 0.0, 'V': 0.0},
          "tn": {'A': 0.0, 'N': 0.0, 'V': 0.0},
          "fn": {'A': 0.0, 'N': 0.0, 'V': 0.0}}

# opens the reference file
with open("data/word-pairs_pos_gt-answer_w2v.tsv") as f:
    # the first line contains the column names, we'll skip it
    f.readline()
    # iterates through each line in the reference file (see exercise1)
    for line in f.readlines():
        word1, word2, pos, issimilar = line.strip().split(column_separator)
        # TODO: convert the value of issimilar to an integer number (hint: see exercise1-inclass)
        issimilar = int(issimilar)

        # compute the cosine similarity using the semantic model
        # by applying the "similarity" method provided by the gensim package
        sim_score = sem_model.similarity(word1, word2)
        # if the model computed a cosine similarity larger than 0.5, 
        # we say that it considers the words as being highly similar in meaning
        if sim_score > 0.5:
            # humans say words are highly similar
            if issimilar == 1:
                print("c", word1, word2, pos, sim_score, issimilar)
                # increase the counter for true positives by 1
                tp += 1 # this is equivalent to tp = tp + 1
                scores["tp"][pos] += 1
            # humans say words are not similar
            # TODO: add condition that prints 'x' in this case
            else:
                print("x", word1, word2, pos, sim_score, issimilar)
                # TODO: increase the counter for false positives by 1
                fp += 1  
                scores["fp"][pos] += 1
                
                
        # model: words are not similar in meaning, but humans: words are highly similar
        # TODO: check if issimilar is 1 (that is, if humans say words are highly similar)
        else:
            # TODO: print "x" and the two words, the pos, the estimated cosine similarity 
            # and the human judgement (see above)
            if issimilar == 1:
                print("x", word1, word2, pos, sim_score, issimilar)    
                # TODO: increase the corresponding counter by 1
                fn += 1
                scores["fn"][pos] += 1
            else:
                print("c", word1, word2, pos, sim_score, issimilar)
                tn += 1
                scores["tn"][pos] += 1
            
            
        # both, model and humans: words are not similar in meaning
        # TODO: go into the "else" case, in which you
        # - print a corresponding statement
        # - increase the corresponding counter
        
        

precision = 0.0
recall = 0.0
accuracy = 0.0


precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)

precision_A = scores['tp']['A'] / (scores['tp']['A'] + scores['fp']['A'])
precision_N = scores['tp']['N'] / (scores['tp']['N'] + scores['fp']['N'])
precision_V = scores['tp']['V'] / (scores['tp']['V'] + scores['fp']['V'])

recall_A = scores['tp']['A'] / (scores['tp']['A'] + scores['fn']['A'])
recall_N = scores['tp']['N'] / (scores['tp']['N'] + scores['fn']['N'])
recall_V = scores['tp']['V'] / (scores['tp']['V'] + scores['fn']['V'])

accuracy_A = (scores['tp']['A'] + scores['tn']['A']) / (scores['tp']['A'] + scores['tn']['A'] + scores['fp']['A'] + scores['fn']['A'])
accuracy_N = (scores['tp']['N'] + scores['tn']['N']) / (scores['tp']['N'] + scores['tn']['N'] + scores['fp']['N'] + scores['fn']['N'])
accuracy_V = (scores['tp']['V'] + scores['tn']['V']) / (scores['tp']['V'] + scores['tn']['V'] + scores['fp']['V'] + scores['fn']['V'])
print("Precision: ", precision_A, precision_N, precision_V)
print("Recall: ", recall_A, recall_N, recall_V)
print("Accuracy: ", accuracy_A, accuracy_N, accuracy_V)
# TODO: compute the precision, recall, and accuracy using the counters tp, fp, tn, fn



            
print("Precison: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)


# OPTIONAL TODO: Evaluate the model on each part-of-speech (nouns (pos=N), adjectives (pos=A), verbs (pos=V)) 
# separately and compare the results across the part-of-speech.
# python hint: adapt the method above, using a dictionary to store the tp, fp, tn, fn for each pos separately, e.g.,
# tp = {"N": 0.0, "A": 0.0, "V": 0.0}
# tp[pos] = tp[pos]+1
# ...

