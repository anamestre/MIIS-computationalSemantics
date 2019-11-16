# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:28:52 2019

@author: Ana Mestre
"""
from nltk.corpus import wordnet as wn

flatten = lambda l: [item for sublist in l for item in sublist]

def read_input(file):
    with open(file) as f:
        input_list = [line.strip() for line in f.readlines()]
    return input_list

def get_hypernyms(word):
    all_synsets = wn.synsets(word)
    hypernyms_list = []
    for synset in all_synsets:
        hypernyms_list.append(synset.hypernym_paths())
    return flatten(hypernyms_list)

def print_results(results):
    filename = "wordnet_categorisation.txt"
    f = open(filename, "w")
    first = True
    for word, category in results:
        if not first:
            f.write("\n")
        f.write(word + "\t" + category)
        first = False
    


categories = read_input("categories_battig.txt")
words = read_input("words_battig.txt")
results = []
for word in words:
    hypernyms = get_hypernyms(word)
    found = False
    for syn_hyp_list in hypernyms: 
        for syn_hyp in syn_hyp_list:
            lemmas_hypernym = syn_hyp.lemma_names()
            for lemma in lemmas_hypernym:
                if lemma in categories:
                    result = [word, lemma]
                    results.append(result)
                    found = True
                    break
            if found:
                break
        if found:
            break
        
                        
print_results(results)