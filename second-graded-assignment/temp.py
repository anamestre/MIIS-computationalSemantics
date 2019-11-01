# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:40:58 2019

@author: Ana
"""

def load_categories_words(words_filepath="wordnet_categorisation.txt"):
    '''Reads a tab-separated file with two columns and returns a dictionary 
    with elements in the first column as keys and elements in the second column as values'''
    my_taxonomy = dict()
    with open(words_filepath) as f:
        for line in f.readlines():
            word, category = line.split()
            my_taxonomy[word] = category
load_categories_words()