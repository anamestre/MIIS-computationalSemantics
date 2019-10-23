# -*- coding: utf-8 -*-
"""
Created 2019

@author: Carina Silberer, Gemma Boleda
"""

import os
import sys

# The Natural Language Toolkit: https://www.nltk.org/
# This toolkit is a standard for linguistic processing of text data -- you will 
# probably work with it in the introductory course on computational linguistics in the second term. 

# If you don't have it, install it with the following command 
# in the Terminal (Mac, Linux) or Anaconda prompt (Windows):
# conda install -c anaconda nltk
import nltk
# *** we use the NLTK's pre-implemented functions for accessing WordNet ***
from nltk.corpus import wordnet as wn

### *** Possible error: ***

#LookupError: 
#**********************************************************************
#  Resource wordnet not found.
#  Please use the NLTK Downloader to obtain the resource:
#
#import nltk
#nltk.download('wordnet')

### this means that, although you have installed NLTK, you don't 
### have the wordnet resource yet.
### *** solution: *** 
## do what the message tells you, in the console:
#import nltk
#nltk.download('wordnet')


### PRACTICE, PART A ###
# In the following, we demonstrate the NLTK commands that you will need to do the exercise.
# Make sure you understand what each command does before you start coding.

print(wn.synsets('can'))
print(wn.synsets('can','n')) # noun synsets
print(wn.synsets('can','v')) # verb synsets
print()

all_synsets = wn.synsets('can','n')
first_synset = all_synsets[0]
print(first_synset)
print()

hypernyms_list=first_synset.hypernym_paths() # ATTENTION! this is a list of list, because each synset can have more than one hypernym.
# In this case, the list has only one element
hypernyms = hypernyms_list[0] # we take the first (and only) hypernym path of this synset
print(hypernyms)
print()

print(first_synset.lemma_names())
print()

## Suggestion: Practice these commands with other synsets of "can", and with words
## from the exercise.


### PRACTICE, PART B ###
# (Uncommend as needed)

## B1. getting hypernyms for more than one word, example:
#
#for word in ["boat", "apple"]:
#   print(word)
#   synsets = wn.synsets(word, 'n')
#   for synset in synsets:
#       print("\tsynset:", synset)
#       list_hypernympaths = synset.hypernym_paths()
#       print("\t\thypernym paths:", list_hypernympaths)
#       print()
       
## B2. going from synsets to words, example:
## (question: is 'artifact' a category of the first synset of 'can'?)

#my_category='artifact'
#for hyp_synset in hypernyms: # remember that we defined variable 'hypernyms' above
#    print(hyp_synset)
#    lemmas_hypernym = hyp_synset.lemma_names()
#    print(lemmas_hypernym) # again, we obtain a list because each synset can have more than one lemma
#    for lemma in lemmas_hypernym:
#        if lemma == my_category:
#            print("******************************")
#            print("*** yes! It's an " + my_category + " ***")
#            print("******************************")
