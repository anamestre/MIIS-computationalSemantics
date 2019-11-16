# -*- coding: utf-8 -*-

import sys, gensim, tsne, nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import matutils 
from nltk.data import find
from matplotlib.pyplot import figure
from sklearn import metrics
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
import pandas as pd

def load_sem_model(sample_model=True):
    if sample_model:
        try:
            word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        except LookupError:
            nltk.download('word2vec_sample')
        word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    else:
        word2vec_modelfname = 'GoogleNews-vectors-negative300.bin.gz'
        try:
            sem_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_modelfname, binary=True)
            print("Google-news model of %d words, each represented by %d-dimensional vectors,    successfully loaded." % (len(sem_model.vocab), sem_model.vector_size))
        except FileNotFoundError:
            sys.stderr.write("Model file with name %s not found in directory. Please download it from %s\n." % (word2vec_modelfname, "https://code.google.com/p/word2vec/"))
            
    vocab_size = len(sem_model.vocab)
    embedding_size = sem_model.vector_size
    print("Model of %d words, each represented by %d-dimensional vectors, successfully loaded." % (vocab_size, embedding_size))
    # We want all vectors to have unit length
    #sem_model.init_sims()
    return sem_model


def plot_with_tsne(vectors, words, color_coding = None, outfile_name="tsne_solution"):
    # Is vectors in the right data structure (numpy array)?
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # Apply t-sne to project the word embeddings into a 2-dimensional space
    Y = tsne.tsne(X=vectors, no_dims=2, initial_dims=int(len(words)/2), perplexity=5.0, max_iter=1000)

    # Let's plot the solution:
    if color_coding is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c = color_coding)
    else:
        plt.scatter(Y[:, 0], Y[:, 1])

    # Let's add the words to the plot:
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=9)

    plt.savefig(outfile_name+".png", format='png', dpi=200)
    plt.show()
    
def plot_with_tsne_kmeans(vectors, words, color_coding_cat, color_coding_kmeans, outfile_name="tsne_solution"):
    # Is vectors in the right data structure (numpy array)?
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
        
    # Apply t-sne to project the word embeddings into a 2-dimensional space
    Y = tsne.tsne(X=vectors, no_dims=2, initial_dims=int(len(words)/2), perplexity=5.0, max_iter=1000)

    # Let's plot the solution:
    if color_coding_cat is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c = color_coding_cat)
    else:
        plt.scatter(Y[:, 0], Y[:, 1])

    # Let's add the words to the plot:
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=9)

    plt.title("Categories distribution")
    plt.savefig(outfile_name+"_cat.png", format='png', dpi=200)
    plt.show()
    
    if color_coding_kmeans is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c = color_coding_kmeans)
    else:
        plt.scatter(Y[:, 0], Y[:, 1])
        
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=9)

    plt.title("Clusters distribution")
    plt.savefig(outfile_name+"_cluster.png", format='png', dpi=200)
    plt.show()


def load_categories_words(words_filepath="wordnet_categorisation.txt"):
    '''Reads a tab-separated file with two columns and returns a dictionary 
    with elements in the first column as keys and elements in the second column as values'''
    my_taxonomy = defaultdict(list)
    categories = []
    words_list = []
    with open(words_filepath) as f:
        for line in f.readlines():
            word, category = line.split()
            words_list.append(word)
            categories.append(category)
            my_taxonomy[word] = category 
    return my_taxonomy, words_list, categories


def word_vec_list(sem_model, relevant_words):
    """ Build a list of word embeddings, vector_list, for the target words 
    given in relevant_words, by means of (1a)-(1c). 
    Since it is possible that not all target words are found in the semantic 
    model, a new list of words, word_list, will be also be created which will 
    only contain those words which are in sem_model.
    """
    word_list = []
    vector_list = []
    for word in relevant_words:
        sem = retrieve_vector(sem_model, word)
        if sem is not None:
            #print("word:", word, sem)
            vector_list.append(sem)
            word_list.append(word)
    return vector_list, word_list


def retrieve_vector(sem_model, word):
    # word is a multi-word-expression -- for simplicity, we assume it's a two-word expression
    if "_" in word:
        word1, word2 = word.split("_", 1)
        vector, _ = compose_words(sem_model, word1, word2)
        return vector
    
    if word not in sem_model:
        print(word, "not found in the semantic model.")
        return None
    return sem_model[word]


def compose_words(sem_model, word1, word2):
    """
    Returns a synthetic vector built by computing 
    the component-wise sum of the embeddings of word1 and word2 and averaging the sum, 
    then normalising this mean vector to unit length (i.e., length == 1, i.e., the dot product is 1: [python command: np.dot(synthetic_vec, synthetic_vec)]).
    """
    if word1 in sem_model and word2 in sem_model:
        vec1 = retrieve_vector(sem_model, word1)
        vec2 = retrieve_vector(sem_model, word2)
        vecs = [vec1, vec2]
        synthetic_vec = matutils.unitvec(np.array(vecs).mean(axis=0)).astype(np.float32)
        #print(type(synthetic_vec))
        return synthetic_vec, word1 + "_" + word2
    else:
        print(word1, "or", word2, "not found in the semantic model.")

    return None, None


def average_vector(sem_model, word_list):
    """
    Returns a synthetic vector built by computing 
    the component-wise sum of the embeddings of the words in word_list and averaging the sum. 
    This mean vector is then normalised to unit length (i.e., length == 1, i.e., the dot product is 1: 
        [python command: np.dot(synthetic_vec, synthetic_vec)]).
    
    This function is a generalisation of compose_words, which expects two words, both contained in the semantic model, 
    to an arbitrary number of words, some of which may not be in the semantic space.
    """
    vectors = []
    words_found = []
    for w in word_list:
        if w in sem_model:
            words_found.append(w)
            sem = retrieve_vector(sem_model, w)
            vectors.append(sem)
    if (len(vectors) > 0):
        synthetic_vec = matutils.unitvec(np.array(vectors).mean(axis = 0)).astype(np.float32)
        return synthetic_vec, words_found

    return None, None


def get_purity(clusters_of_words, reference_clusters):
    sum_overlap = 0
    contingency_matrix = metrics.cluster.contingency_matrix(reference_clusters, clusters_of_words)
    print(contingency_matrix)
    sum_overlap = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return sum_overlap

def invert_dict(dictionary):
    new_dictionary = defaultdict(list)
    for word in dictionary.keys():
        new_dictionary[dictionary[word]].append(word)
    return new_dictionary


def cosine_similarity(A, B):
    dot_val = dot(A, B)
    lens_val = norm(A) * norm(B)
    return dot_val / lens_val

def print_cosine_similarity(sem_model, cat_name, pro_category, hyponyms):
    hyper_name = cat_name
    hypernym = retrieve_vector(sem_model, hyper_name)
    if hypernym is not None:
        print("--- Cosine similarities for hypernym:", hyper_name.upper())
        if pro_category is not None:
            cos_sim = cosine_similarity(hypernym, pro_category)
            print("------ with prototype category:", cat_name.upper(), ", cos sim:", cos_sim)
        
        for hypo in hyponyms:
            vec = retrieve_vector(sem_model, hypo)
            if vec is not None:
                cos_sim = cosine_similarity(hypernym, vec)
                print("------ with hyponym:", hypo.upper(), ", cos sim:", cos_sim)

def get_colors(kmeans_labels, cat_labels):
    # Color vector for clusters
    color_vector = []
    for label in kmeans_labels:
        if label == 0:
            color = "aquamarine"
        elif label == 1:
            color = "coral"
        elif label == 2:
            color = "darkgreen"
        elif label == 3:
            color = "crimson"
        elif label == 4:
            color = "lavender"
        elif label == 5:
            color = "chocolate"
        elif label == 6:
            color = "fuchsia"
        elif label == 7:
            color = "black"
        elif label == 8:
            color = "khaki"
        elif label == 9:
            color = "violet"
        elif label == 10:
            color = "olive"
        color_vector.append(color)
        
    # Color vector for categories
    color_basic = []
    for label in cat_labels:
        if label == "mammal":
            color = "blue"
        elif label == "bird":
            color = "orange"
        elif label == "fish":
            color = "green"
        elif label == "vegetable":
            color = "red"
        elif label == "fruit":
            color = "purple"
        elif label == "tree":
            color = "brown"
        elif label == "vehicle":
            color = "pink"
        elif label == "clothing":
            color = "grey"
        elif label == "tool":
            color = "yellowgreen"
        elif label == "kitchen_utensil":
            color = "turquoise"
        elif label == "dishware":
            color = "navy"
        color_basic.append(color)
    return color_vector, color_basic
        

# ------ Initializations
sem_model = load_sem_model()
taxonomy, all_words, categories = load_categories_words("wordnet_categorisation.txt") # Dict[word] = [categories]
tokens_to_plot, token_colours, vectors_to_plot = [], [], []
category_to_words = invert_dict(taxonomy)
categories = category_to_words.keys()


# ------ Words
vectors_to_plot_w, tokens_to_plot_w = word_vec_list(sem_model, all_words) # There are no None values here
token_colours_w = ["blue"]*len(tokens_to_plot_w)
tokens_to_plot_w = ['']*len(tokens_to_plot_w)


# ------ Hypernyms
vectors_to_plot_c, tokens_to_plot_c = word_vec_list(sem_model, categories)
token_colours_c = ["magenta"]*len(tokens_to_plot_c)

vectors_to_plot = vectors_to_plot_w + vectors_to_plot_c
tokens_to_plot = tokens_to_plot_w + tokens_to_plot_c
token_colours = token_colours_w + token_colours_c
        

## This should plot words and categories:
figure(num = None, figsize = (10, 10), dpi = 80, facecolor = 'w', edgecolor = 'k')
plot_with_tsne(vectors_to_plot, tokens_to_plot, color_coding = token_colours, outfile_name="vis_cats_words_coloured")


# ------ Categories
for (category, words) in category_to_words.items():
    average, tokens = average_vector(sem_model, words) # Prototype category vector
    print_cosine_similarity(sem_model, category, average, tokens)
    # for plotting prototype categories
    if average is not None and tokens is not None:
        tokens_to_plot.append(category.upper())
        token_colours.append("yellow")
        vectors_to_plot.append(average)


# ------ Plot with tsne: Words, Hypernyms & Categories
figure(num = None, figsize = (10, 10), dpi = 80, facecolor = 'w', edgecolor = 'k')
plot_with_tsne(vectors_to_plot, tokens_to_plot, color_coding = token_colours, outfile_name = "vis_cats_words_prototypes_coloured")


# ------ Kmeans clustering
num_clusters = 11
vectors_to_cluster_, words_to_cluster = word_vec_list(sem_model, all_words)
vectors_to_cluster = np.array(vectors_to_cluster_)
kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(vectors_to_cluster)

# ------ Get real categories
cat_cluster = []
for w_cluster in words_to_cluster:
        cat_cluster.append(taxonomy[w_cluster])


cluster_to_words = defaultdict(list)
categories_in_clusters = defaultdict(lambda: defaultdict(int))
for (w, cluster_w) in zip(words_to_cluster, kmeans.labels_):
    cluster_to_words[cluster_w].append(w)
    categories_in_clusters[taxonomy[w]][cluster_w] += 1


# ------ Barplot representing clusters and categories
figure(num = None, figsize = (10, 10), dpi = 80, facecolor = 'w', edgecolor = 'k')
df = pd.DataFrame(categories_in_clusters)
df.plot(kind="bar", stacked=True)
plt.show()


# ------ Print the words in each cluster. 
for cluster in cluster_to_words.keys():
    print("--- Cluster", cluster, "---")
    for word in cluster_to_words[cluster]:
        print(word)


# ------ Plot categories and clusters
color_vector, color_basic = get_colors(kmeans.labels_, cat_cluster)
tokens_cluster = ['']*len(words_to_cluster)
plot_with_tsne_kmeans(vectors_to_cluster_, tokens_cluster, color_basic, color_vector, outfile_name = "kmeans")
 

# ------ Compute purity:
purity_score = get_purity(kmeans.labels_, cat_cluster)
print("Purity: %.3f"%purity_score)