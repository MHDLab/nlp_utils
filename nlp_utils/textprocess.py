
import re
import os
import pandas as pd
from tqdm import tqdm

import spacy
import string

from difflib import SequenceMatcher
from nltk.corpus import wordnet

pd.options.mode.chained_assignment = None
punctuations = string.punctuation
nlp = spacy.load('en_core_sci_lg')

#Lexicon normalization and noise removal; an importable, multisituational function for normalization
def tokenizer(text, lookup_dict=None, custom_stop_words=[], skip_words=[], custom_autocorrect=None):
    mytokens = nlp(text)

    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in punctuations]

    if len(custom_stop_words):
        mytokens = [word for word in mytokens if word not in custom_stop_words]

    mytokens = " ".join([i for i in mytokens])
    mytokens = re.sub('\W+',' ', mytokens)
    mytokens = re.sub('\s+',' ', mytokens)

    if lookup_dict != None:
        mytokens = _standardize_words(mytokens, lookup_dict)

    
    if custom_autocorrect != None:

        mytokens = mytokens.split()

        mytokens1 = []

        corrected_words = 0
        for token in mytokens:
            syns = wordnet.synsets(token)
            if not syns and token not in skip_words:
                token1 = token
                token = return_correct(token, custom_autocorrect)
                if token != "Null":
                    # print("this token: (%s) has been corrected into: (%s)", token1, token)
                    mytokens1.append(token)
                    corrected_words += 1
                else:
                    mytokens1.append(token1)
            else:
                mytokens1.append(token)

        print("Amount of corrected words are: ", corrected_words)

        mytokens1 = " ".join([i for i in mytokens1])
        if lookup_dict != None:
            mytokens1 = _standardize_words(mytokens1, lookup_dict)

        mytokens = mytokens1

    return mytokens


#Damerau-LV Distance outputs minimum cost of converting on string into another utilizing
#insertions, deletions, substitutions, and transpositions
def damerau_levenshtein(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            #recursively called functions:
            #indicator function
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1

            d[(i,j)] = min(d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )

            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def return_correct(word, custom_autocorrect):

    correct_string = list()
    #Creating a dictionary to store each word from the loaded dictionary and its distance from the input word
    dict = {}
    for x in custom_autocorrect:
        dict.update({x : damerau_levenshtein(x,word)})

    #Sorting dictionary by value, from smallest to largest distance
    sorted_d = sorted(dict.items(), key=lambda x: x[1])
    correct_string.append(sorted_d[0][0])

    if SequenceMatcher(None, sorted_d[0][0], word).ratio() >= 0.80:
        # print("This word has been corrected: ", word)
        return sorted_d[0][0]
    else:
        # print(SequenceMatcher(None, sorted_d[0][0], word).ratio())
        # print(word, "is none")
        return "Null"

#Function to load dictionary with custom entries


#Object standardization; an importable, multisituational function for standardize objects.
#Dictionary must be loaded using set_dictionary prior to calling
def _standardize_words(input_text, lookup_dict):
    words = input_text.split()
    # print(type(words))
    new_words = []
    for word in words:
        if word in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
    new_text = " ".join(new_words)
    return new_text

# flagging words that are not in Wordnet dictionary or in skip_words
def flag(text, custom_stop_words, skip_words):
    removed_words = 0
    kept_words = 0
    lst = text.split()
    new_text = []
    mispelled = []
    for element in lst:
        syns = wordnet.synsets(element)
        if not syns and element not in skip_words:
            removed_words += 1
            mispelled.append(element)
            continue
        elif element.isnumeric():
            removed_words += 1
            mispelled.append(element)
            continue
        elif syns and element in custom_stop_words:
            removed_words += 1
            mispelled.append(element)
            continue
        else:
            kept_words += 1
            new_text.append(element)
    print("Amount of removed words: ", removed_words)
    print("Amount of kept words: ", kept_words)

    misspelled = " ".join([i for i in mispelled])
    new_text = " ".join([i for i in new_text])
    return new_text, misspelled
