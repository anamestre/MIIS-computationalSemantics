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
    
    
    # -------------------------------------------------------------------
    # ----------------------- TASKS -------------------------------------
    # -------------------------------------------------------------------
    
    words = ["apple", "sweet", "allow"]
    
    # a) 20 nearest neighbors of the words in a space built on the google news-gram corpus, using
    # the word2vec semantic model.
    
    for w in words:
        neighbors = sem_model.most_similar(w, topn = 20)
        print("For the word " + w + " here are the 20 nearest neighbors and its similarity score: ")
        for n in neighbors:
            print("  - Word: " + n[0] + ", Similarity score: " + str(n[1]))
    """
    RESULTS:
    For the word apple here are the 20 nearest neighbors and its similarity scores: 
      - Word: apples, Similarity score: 0.720359742641449
      - Word: pear, Similarity score: 0.6450697183609009
      - Word: fruit, Similarity score: 0.641014575958252
      - Word: berry, Similarity score: 0.6302294135093689
      - Word: pears, Similarity score: 0.6133961081504822
      - Word: strawberry, Similarity score: 0.6058261394500732
      - Word: peach, Similarity score: 0.6025872230529785
      - Word: potato, Similarity score: 0.5960935354232788
      - Word: grape, Similarity score: 0.5935864448547363
      - Word: blueberry, Similarity score: 0.5866668224334717
      - Word: cherries, Similarity score: 0.5784382224082947
      - Word: mango, Similarity score: 0.5751855373382568
      - Word: apricot, Similarity score: 0.5727777481079102
      - Word: melon, Similarity score: 0.5719985365867615
      - Word: almond, Similarity score: 0.5704830288887024
      - Word: Granny_Smiths, Similarity score: 0.5695333480834961
      - Word: grapes, Similarity score: 0.5692256093025208
      - Word: peaches, Similarity score: 0.5659247040748596
      - Word: pumpkin, Similarity score: 0.5651882886886597
      - Word: apricots, Similarity score: 0.5645568370819092
    
    
    For the word sweet here are the 20 nearest neighbors and its similarity score: 
      - Word: sweetness, Similarity score: 0.6216813325881958
      - Word: sweetest, Similarity score: 0.6207044720649719
      - Word: caramelly, Similarity score: 0.6009937524795532
      - Word: syrupy_sweet, Similarity score: 0.5979651212692261
      - Word: yummy, Similarity score: 0.5963823795318604
      - Word: buttery, Similarity score: 0.593994140625
      - Word: tooth_achingly, Similarity score: 0.5892406105995178
      - Word: fan_Rosangela_Pereira, Similarity score: 0.5891239047050476
      - Word: delicious, Similarity score: 0.5866374969482422
      - Word: fruity, Similarity score: 0.5804054737091064
      - Word: unsweet, Similarity score: 0.5799071788787842
      - Word: Daisuke_Matsuzaka_Dustin_Pedroia, Similarity score: 0.5718575716018677
      - Word: peanutty, Similarity score: 0.5682952404022217
      - Word: tasty, Similarity score: 0.5673485398292542
      - Word: perfumey, Similarity score: 0.5672965049743652
      - Word: lovely, Similarity score: 0.5654473304748535
      - Word: chocolaty, Similarity score: 0.5646396279335022
      - Word: tangy_sweet, Similarity score: 0.5615671873092651
      - Word: grapey, Similarity score: 0.5592212677001953
      - Word: tangy, Similarity score: 0.5577353239059448
    
    For the word allow here are the 20 nearest neighbors and its similarity score: 
      - Word: enable, Similarity score: 0.749002993106842
      - Word: allowing, Similarity score: 0.6995344161987305
      - Word: allows, Similarity score: 0.6941528916358948
      - Word: allowed, Similarity score: 0.6417946815490723
      - Word: require, Similarity score: 0.616129457950592
      - Word: enabling, Similarity score: 0.5952922701835632
      - Word: enabled, Similarity score: 0.5854588747024536
      - Word: enables, Similarity score: 0.5658420324325562
      - Word: let, Similarity score: 0.556652843952179
      - Word: Allow, Similarity score: 0.5466748476028442
      - Word: requiring, Similarity score: 0.5461660027503967
      - Word: lets, Similarity score: 0.5376736521720886
      - Word: encourage, Similarity score: 0.5248451232910156
      - Word: Allowing, Similarity score: 0.5220657587051392
      - Word: facilitate, Similarity score: 0.5159643292427063
      - Word: restrict, Similarity score: 0.5144646763801575
      - Word: compel, Similarity score: 0.5121440887451172
      - Word: permitted, Similarity score: 0.511229395866394
      - Word: Allows, Similarity score: 0.4960828125476837
      - Word: give, Similarity score: 0.49367666244506836

    """
    
    # b) 10 nearest neighbors for two-word phrases that you associate to different senses of the 
    # word (e.g., for "mouse", "use mouse" vs. "furry mouse").
    two_words = ["big", "apple"]
    res_two_words = sem_model.most_similar(two_words, topn = 10)
    print("The 10 nearest neighbors for the two-word phrase " + two_words[0] + " " + two_words[1] + " are:")
    for w, s in res_two_words:
        print("  - Word: " + w + ", Score similarity: " + str(s))
    
    two_words2 = ["tasty", "apple"] 
    res_two_words2 = sem_model.most_similar(two_words2, topn = 10)
    print("The 10 nearest neighbors for the two-word phrase " + two_words2[0] + " " + two_words2[1] + " are:")
    for w, s in res_two_words2:
        print("  - Word: " + w + ", Score similarity: " + str(s))
    
    """
    RESULTS:
    The 10 nearest neighbors for the two-word phrase big apple are:
      - Word: huge, Score similarity: 0.576988160610199
      - Word: bigger, Score similarity: 0.5484350919723511
      - Word: apples, Score similarity: 0.5447474718093872
      - Word: pumkin, Score similarity: 0.5325223803520203
      - Word: gigantic, Score similarity: 0.5123871564865112
      - Word: levee_Boudrie, Score similarity: 0.509497880935669
      - Word: mango_tango, Score similarity: 0.5072993040084839
      - Word: humongous, Score similarity: 0.4980980455875397
      - Word: potato, Score similarity: 0.4962328374385834
      - Word: plump_de_manzana, Score similarity: 0.4866654872894287
    
    The 10 nearest neighbors for the two-word phrase tasty apple are:
      - Word: delicious, Score similarity: 0.7507026195526123
      - Word: yummy, Score similarity: 0.7134654521942139
      - Word: tart_apple, Score similarity: 0.6985276341438293
      - Word: crunchy_salty, Score similarity: 0.698421061038971
      - Word: juicy_watermelon, Score similarity: 0.689346194267273
      - Word: blackberry_pie, Score similarity: 0.6826300621032715
      - Word: scrumptious, Score similarity: 0.6767407655715942
      - Word: bing_cherries, Score similarity: 0.6725069284439087
      - Word: juicy_peaches, Score similarity: 0.6700095534324646
      - Word: berry_pies, Score similarity: 0.6696867942810059

    """

    
    # c) Cosine similarity between the word and:
    # - one (near-)synonym, ---> fruit, sugary, concede
    # - one topically-related word,  ---> pear, sour, acknowledge
    # - one hypernym,---> fruit, flavour, provide
    # - one antonym (if available)  ---> salty, forbid
    synonyms = ["fruit", "sugary", "concede"]
    topic_related = ["pear", "sour", "acknowledge"]
    hypernyms = ["fruit", "flavor", "provide"]
    antonyms = ["salty", "forbid"]
    
    print(" ---- Synonyms ----")
    res_synonyms = map(lambda x,y: sem_model.similarity(x, y), words, synonyms)
    for w, s, r in zip(words, synonyms, res_synonyms):
        print("For the word " + w + " the cosine similarity to " + s + " is " + str(r))
    
    print(" ---- Topic related ----")
    res_topic_related = map(lambda x,y: sem_model.similarity(x, y), words, topic_related)
    for w, s, r in zip(words, topic_related, res_topic_related):
        print("For the word " + w + " the cosine similarity to " + s + " is " + str(r))
    
    print(" ---- Hypernyms ----")
    res_hypernyms = map(lambda x,y: sem_model.similarity(x, y), words, hypernyms)
    for w, s, r in zip(words, hypernyms, res_hypernyms):
        print("For the word " + w + " the cosine similarity to " + s + " is " + str(r))
    
    print(" ---- Antonyms ----")
    res_antonyms = map(lambda x,y: sem_model.similarity(x, y), words[1:], antonyms)
    for w, s, r in zip(words[1:], antonyms, res_antonyms):
        print("For the word " + w + " the cosine similarity to " + s + " is " + str(r))
    
    """
    RESULTS:
     ---- Synonyms ----
    For the word apple the cosine similarity to fruit is 0.6410147
    For the word sweet the cosine similarity to sugary is 0.51119614
    For the word allow the cosine similarity to concede is 0.18384954
     ---- Topic related ----
    For the word apple the cosine similarity to pear is 0.6450697
    For the word sweet the cosine similarity to sour is 0.41820753
    For the word allow the cosine similarity to acknowledge is 0.21194625
     ---- Hypernyms ----
    For the word apple the cosine similarity to fruit is 0.6410147
    For the word sweet the cosine similarity to flavor is 0.39978844
    For the word allow the cosine similarity to provide is 0.49169135
     ---- Antonyms ----
    For the word sweet the cosine similarity to salty is 0.49490488
    For the word allow the cosine similarity to forbid is 0.4009567
    
    OBSERVATIONS:
        * The cosine similarity between antonyms is pretty high because of their topic-relateness.
        * The cosine similarity is quite high between synonyms and topically related words except for "allow" which
        is a tricky word to find a synonym or a a topic-related word.
    """
