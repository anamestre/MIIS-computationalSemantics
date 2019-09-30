#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:14:03 2019

@author: Carina Silberer

Course: Computational Linguistics
Assignment Week 1: Distributional Semantics
"""
import math
import sys

""" 
Task 1: Calculus (addition and multiplication)
"""
print("\nTASK 1")

# a) Calculation
print("\na) Calculation:")
# Given two numbers, a and b
a = 2
b = 4
# .. print the sum of a and b
print("\n.. print the sum of a and b:")

# .. sum a and b and assign the result to a variable called sum_ab, print sum_ab
print("\n.. sum a and b and assign the result to a variable called sum_ab, print sum_ab:")

# .. multiply a and b and assign the result to a variable called prod_ab, print prod_ab
print("\n.. multiply a and b and assign the result to a variable called prod_ab, print prod_ab:")

# .. print the square root of number 9
# .. python hint: math.sqrt() function
# .. this is a module and we need to import it (see 'import math' above)
print("\n.. print the square root of number 9:")

""" 
Task 2: Defining and using a new function
.. python background to "Adding new functions":
   https://www.py4e.com/html3/04-functions
"""
print("\nTask 2: Defining and using a new function")
var = "you"
# Understand the following print statement
print("0) Understand the following print statement:")
print("Hello", var)

# a) Define a function, called 'greeting' that prints 'Hello'
print("\na) Define a function, called 'greeting' that prints 'Hello':")
    
# .. add another line to the function body, that prints 'How are you?'
print("\n.. add another statement to the function body, that prints 'How are you?':")
    
# .. Call the function

# b) Modify the function, such that it takes an argument called 'a_name'
print("\nb) Modify the function, such that it takes an argument called 'a_name'")
    
# .. now modify the function body, such that it prints the value of 
# its parameter 'a_name' just behind the 'Hello'
print("-- modify the function body, such that it prints the value of its parameter 'a_name' just below the 'Hello'")

# .. call the function 2 times, giving it once above variable 'var' as argument, and once your name, e.g.:
print("-- and call the function 2 times, giving it once above variable 'var' as argument, and once your name:")
# greeting(var)
# greeting("Carina")


""" 
Task 3: Operate on lists
"""
print("\nTASK 3")
# a) Given a list with strings and numbers, print the first element in the list
print("\na) Given a list with strings and numbers,")
some_list = ["first_element", 1, 2, 3]
print(some_list)

print(".. print the first element in the list:")
# note: we start counting by 0, so the first element is in position 0

# b) Given a list with strings and numbers, iterate over its elements and print them one after the other
# .. python hint: for-loop
print("\nb) Given the same list, iterate over its elements and print them one after the other:")
# in the following list, make sure you know why "first elements is in quotation marks, while the numbers are not; https://www.py4e.com/html3/02-variables
some_list = ["first_element", 1, 2, 3] 

# c) Given a list with strings and numbers, print a sublist of all its elements but the first one
# .. python hint: list slices
print("\nc) Given the same list, print a sublist of all its elements but the first one:")

# d) Given a list with strings and numbers, starting from the second element (i.e., index 1), 
# print all its elements, one after the other
# .. python hint: list slices, for-loop
print("\nd) Given the same list, starting from the second element (i.e., index 1), print all its elements, one after the other:")
    
# e) Store all elements of the list BUT the first one (i.e., all elements starting from index 1) in a new list
# .. python hint: e.g., create a new, empty list and use append (see lists)
print("\ne) Store all elements of the list BUT the first one (i.e., all elements starting from index 1 in a new list:")
new_list = []

""" 
Task 4: String operations with lists
"""
print("\nTASK 4")
a_list = [2, "23"]
# a) Given a list with a number and a string number, print the list
print("\na) Given a list with a number and a string number, print the list:")

# b) Given the same list, convert the string number to a number
# .. python hint: float()
print("\nb) Given the same list, convert the element that contains a number in string format to a number with a decimal point (float type):")
# python background: https://www.py4e.com/html3/02-variables

""" 
Task 5: Calculus with vectors (lists of numbers)
"""
print("\nTASK 5")
#
# a) Given a list of numbers (a vector)
print("\na) Given a list of numbers (a vector):")
some_numbers = [2, 2, 4]
print(some_numbers)

# .. sum element 2 and 3 of the list and print the result
print("\n.. sum element 2 and 3 of the list and print the result:")

# b) Print the length of the list some_numbers
# .. python hint: len() function
print("\nb) Print the length of the list some_numbers:")

# c) What does the function range() do?
print("\nc) ADVANCED Question: What does the function range() do?:")
# python hint: check tuples in the book
print(range(5))

# d) What does the function range() do, part 2?
print("\nj) Question: What does the function range() do, part 2?")
for i in range(5):
    print(i)

# e) Understand what the following two lines do:
print("\nk) Question: What do the following two lines do?")
for index in range(len(some_numbers)):
    print(some_numbers[index])

# f) What do the following lines do:
print("\nl) Question: What do the following lines do?")
result = 0
for index in range(len(some_numbers)):
    result = result + some_numbers[index]
    
print(result)

# g) Alternative to f:
print("\n.. Alternative to f) (code):")
result = 0
for num in some_numbers:
    result = result + num

print(result)

""" 
Task 6: Open a file, read and print its content
.. python background to opening and reading in files: 
   https://www.py4e.com/html3/07-files
"""
print("\nTASK 6")
# Given a file with the name "toy-matrix-raw.txt"
filename = "toy-matrix-raw.txt"
print(filename)

# a) Read in the whole content and print it
print("\na) Open the file, read in the whole content and print it:")

# b) Only print the first line
print("\nb) Open the file, and only print the first line:")

# c) Assign the content of the first line to a variable, called "line", 
#    and print it
print("\nc) Open the file, assign the content of the first line to a variable, called 'line' and print it:")

# d) Strip off the spaces at the beginning and end of the line
# .. python background:
#   https://www.py4e.com/html3/06-strings
print("\nd) In addition to c), strip off the spaces at the beginning and end of the line:") 

# e) In addition to stripping off the spaces, split the line by white space ==> this returns a list
# .. python hint and background:
#   use a string method for splitting
#   https://www.py4e.com/html3/08-lists
print("\ne) ... In addition to d) (stripping off the spaces), split the line by white space:")


# f) Read in all the lines of the file and print them (as a list of lines)
print("\nf) Read in all the lines of the file and print them (as a list of lines):")

# g) Read in all the lines of the file, print one after the other
# .. python hint: for-loop
#    https://www.py4e.com/html3/05-iterations
print("\ng) Read in all the lines of the file, print one after the other:")

"""
Task 7: Dictionaries
.. python background:
   https://www.py4e.com/html3/09-dictionaries
"""
print("\nTASK 7")
# a) Given a dictionary with two entries, 
print("\na) Given a dictionary with two entries,")
a_dict = {"apple": [1, 2, 2],
          "cherry": [3, 3, 3]
          }
print(a_dict)

# ... print the value stored under key "apple"
print("\n... print the value stored under key 'apple':")

# a) Given the same dictionary and a list, 
print("\na) Given the same dictionary and a list,")
list_of_nums = [4, 4, 4]
print(list_of_nums)
# ... add another entry, with the key "pear" and the list_of_nums as value
print("\n... add another entry, with the key 'pear' and the list_of_nums as value:")

# ... and print the dictionary
print("\n... and print the dictionary:")


""" 
Task 8: Measuring the cosine similarity between two vectors
See Equation 6.10, SLP3, Chapter 6.4
"""
print("\nTASK 8: Measuring the cosine similarity")
# Given two vectors (i.e., lists of numbers)
vector1 = [1, 2, 3]
vector2 = [3, 3, 3]

##############################
# a) This is an exercise on paper (NO COMPUTER USE)

# Compute the SQUARED SUM of vector1, i.e., sum the product of each element in vector1 with itself
# .. Do the same with vector2

# Compute the LENGTH of vector1, i.e., take the square root of the squared sum of vector1
# .. Do the same with vector2

# Compute the product of (i.e., multiply) the length of vector1 and the length of vector2

# Compute the DOT PRODUCT of vector1 and vector2, i.e., multiply the elements in vector1 with the corresponding 
# elements in vector2 (element at same positions), and compute the sum of that

# Compute the COSINE SIMILARITY of vector1 and vector2, i.e., 
# divide their dot product
# by the product of their lengths
##############################

# Now use python for the following subtasks
# If you have solved all the Tasks above, you have all the tools you need!

# a) Iterate over the elements in vector1, print the product (multiplication) 
# of each element with itself
# e.g., vector1[2]*vector1[2] = 9
# .. python hint: for-loop
        
# b) SQUARED SUM: Sum the product of each element in vector1, store the result in variable "prod1"
# this would mean that prod1 contains the result of 1*1 + 2*2 + 3*3
prod1 = 0
# CODE GOES HERE
    
print(prod1)

# c) LENGTH of a vector: Compute the square root of the squared sum of vector1, 
# and assign the result to variable length1, print length1

# d) DOT PRODUCT: Multiply the elements in vector1 with the corresponding 
# elements in vector2 (element at same positions), and compute the sum of that
dot_prod = 0.0
# CODE GOES HERE

print(dot_prod)

# e) Combining subtasks: Do you see that you iterate over the same list of numbers (the indices of vector1) in each Subtask c-d?
# Let's make this a bit more efficient:
# Just do the iteration (for-loop) once, and do all the steps in the body of that for-loop
prod1 = 0
length1 = 0.0
dot_prod = 0.0


# f) Completing loop body by adding vector2 
# In order to compute the cosine similarity between two vectors, 
# it is required that both have the same number of elements, i.e., len(vector1) == len(vector2), and thus range(len(vector1)) == range(len(vector2))
# Extend your code under e) such as it also computes the length of *vector2*
prod1 = 0
prod2 = 0
length1 = 0.0
length2 = 0.0
dot_prod = 0.0
# CODE GOES HERE

print(prod1, prod2, dot_prod)

# g) Final step to measure the cosine similarity: Divide the dot product of vector1 and vector2 by the product of their lengths. 
# The result should be 0.9258200997725514

