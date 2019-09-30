#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:14:03 2019

@author: Carina Silberer

Course: Computational Linguistics
Assignment Week 1: Distributional Semantics

OVERVIEW:
The goal of this assignment is to measure the cosine similarity between word vectors. 
The vectors are stored in a text file and need to be loaded and preprocessed. 
Then you need to compute the cosine similarity between two vectors.
In order to do this for all pairs of vectors, it is more efficient to define a function that computes the cosine similarity, given two vectors. 
You can then apply the function on any two vectors (i.e., words), and compare the resulting similarity scores between pairs of words, such as the similarity between 'fruit' and 'apple' and 'animal' and 'apple'.


INSTRUCTIONS:
Work through this script, and solve all the exercises. More specifically, your task is to fill in the holes in the code, as indicated by '# TODO' above the corresponding hole.

Example:
--------
some_vec = [1, 2, 3, 4]
# TODO: print the second element of the list (vector) some_vec

--------
===>
--------
some_vec = [1, 2, 3, 4]
# TODO: print the second element of the list (vector) some_vec
print(some_vec[1])
--------
"""


import math
import sys


""" 
TASK 1: Implementation
Step 1: 
Goal (substeps follow below): Open the text file "short-cooccurrence-matrix-raw.txt" 
which contains the shortened co-occurrence matrix of SLP3, Chapter 6.4 (page 12)
and read in the word vectors
File format:
    the first line contains the context words
    each of the remaining lines corresponds to one word vector, where 
    - the first column gives the target word, and 
    - each number gives the co-occurrence weight for a particular context word
"""
filename = "short-cooccurrence-matrix-raw.txt"

#####
## Open the file and read its content, one line after the other, 
# and store the vectors in a dictionary (called 'vectors'), 
# keys: target words, values: list of numbers (the vector)
# .. python background:
#   https://www.py4e.com/html3/09-dictionaries
vectors = dict()

# .. python background to opening and reading in files: 
#   https://www.py4e.com/html3/07-files
with open(filename) as f:
    # read in the first line which contains the context words - we don't need it though
    context_words = f.readline()
    # iterate over the remaining lines in the file, and do:
    for remaining_line in f.readlines():
        # TODO: remove (strip) the white space at the beginning and end of the line, 
        # then split it by white space (==> into the target word and the individual numbers)
        # store the result in "target_vector" (i.e., assignment statement)
        # .. python background:
        #   https://www.py4e.com/html3/06-strings
        print("TO BE IMPLEMENTED BY YOU")
        
        
        # converts each number to a float number and stores the result in "vector"
        # UNCOMMENT (i.e., remove '#') THE FOLLOWING LINE ONCE YOU HAVE DONE THE TODO ABOVE:
        #vector = [float(num) for num in target_vector[1:]]
        
        # TODO: store the target word (i.e., the first element in "target_vector")
        # in a variable called "target"
        # .. python background:
        #   https://www.py4e.com/html3/08-lists
        print("TO BE IMPLEMENTED BY YOU")
        
        
        # TODO: add the vector into the dictionary "vectors"
        # .. python background:
        #   https://www.py4e.com/html3/09-dictionaries
        print("TO BE IMPLEMENTED BY YOU")
        

# print the individual target words and their vectors to check whether everything has been processed correctly:
for word in vectors:
    print("Vector of " + word + ": " + str(vectors[word]))

print("")

"""
Step 2: Compute the cosine similarity between two words
See Equation 6.10, SLP3, Chapter 6.4
"""
v = "cherry"
w = "information"
# Test whether words are in vectors are retrieve their vector representations if they are
if v in vectors and w in vectors:
    print("Vector of " + v + ": " + str(vectors[v]))
    print("Vector of " + w + ": " + str(vectors[w]))
    # Stores the vectors of 'cherry' and 'information' in variables vector_v and vector_w
    vector_v = vectors[v]
    vector_w = vectors[w]

cosine_sim = 0.0
dot_product = 0.0
length_v = 0.0
length_w = 0.0
# TODO: compute the cosine similarity between vector_v and vector_w (i.e., word v and w)
print("TO BE IMPLEMENTED BY YOU")

print("Words: " + v + " and " + w)
print("Their cosine similarity is: ")
print(cosine_sim)
print("")

"""
Step 3: We'll put above code in a function body, such that we can use it easily also 
for other pairs of words
.. python background to "Adding new functions":
  https://www.py4e.com/html3/04-functions
"""
def cosine_similarity(v, w):
    # cosine_similarity computes the cosine similarity between word v and w
    # TODO: Put the code of Step 2 into the function body
    print("TO BE IMPLEMENTED BY YOU")

# call the function with v and w as arguments, store the value it returns into 
# the variable "csim"
csim = cosine_similarity(v, w)
# print the computed similarity (i.e., csim)
print("Words: " + v + " and " + w)
print("Their cosine similarity is: ")
print(csim)

print("") 

v = "digital"
w = "information"
print("Words: " + v + " and " + w)
print("Their cosine similarity is: ")
print(cosine_similarity(v, w))

"""
Task 2: Analysis, using your implementation
"""
filename = "cooccurrence-matrix-raw.txt"
# TODO: Now read in the larger matrix "cooccurrence-matrix-raw.txt", and 
# compute the cosine similarity between *each pair* of words
# Example: Call cosine_similarity("cherry", "strawberry"), and print the returned result
print("TO BE IMPLEMENTED BY YOU")

# Describe your observations - which word pair has the highest similarity, which one has the lowest?


