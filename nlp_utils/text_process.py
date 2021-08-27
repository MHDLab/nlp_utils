"""
https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py
Seems like gensim convention is 

documents: single string for each document
texts: list of string words for each document
corpus: digitized bag of words for
"""

def stopword_removal(text, stopwords):
    text_out = []
    for word in text:   
        if word.lower() not in stopwords:
            text_out.append(word)
    
    return text_out

def apply_fn_text(text, fn, **kwargs):
    """applies a function to each word of a text (list of words)"""
    text_out = []

    for word in text:
        word = fn(word, **kwargs)
        text_out.append(word)     

    return text_out

import sys
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
sys.path.append(r'C:\Users\aspit\Git\MLEF-Energy-Storage\ES_TextData\mat2vec')
from mat2vec.processing import MaterialsTextProcessor
from . import text_analysis

def text_processing_pipeline(docs, debug = False):

    #TODO: combine with punctuation below. Found it would only get isolated punct, not like "however,"
    for punct in string.punctuation.replace('-',''):
        docs = docs.apply(lambda x: x.replace(punct,''))

    docs = docs.apply(lambda x: x.replace('-', '_'))

    #TODO: figure out encoding...this is a 'long hyphen'
    #https://stackoverflow.com/questions/19149577/python-replace-long-dash-with-short-dash
    docs = docs.apply(lambda x: x.replace(b'\xe2\x80\x93'.decode('utf-8'), '_'))

    texts = docs.apply(str.split)

    print('Removing stopwords')
    en_stopwords  = stopwords.words('english')
    all_stops = set(en_stopwords) | set(string.punctuation)

    texts = texts.apply(lambda x: stopword_removal(x, all_stops))

    if debug: print([w[0] for w in text_analysis.top_words(texts, num_words=30)])


    print('Running Mat2Vec text processing')
    # Basic code for implementing mat2vec. need to initialize the mat2vec repo git submodule in the ES_TextData folder. This replaces numbers with the string '<nUm>' which I have not implemented code to remove. 
    materials_text_processor = MaterialsTextProcessor()

    texts = texts.apply(
        lambda x: materials_text_processor.process(x, exclude_punct=True, replace_elements = True)[0]
    )

    # mat2vec replaces numbers with '<nUm>', so remove this from the texts. 
    texts = texts.apply(lambda x: [t for t in x if t != '<nUm>'])

    if debug: print([w[0] for w in text_analysis.top_words(texts, num_words=30)])

    print('Performing Lemmatization')

    #Use lemmatizer to keep words readable and potentially improve modeling. At the cost of speed.
    wn_lemmatizer = WordNetLemmatizer()

    texts = texts.apply(lambda x: apply_fn_text(x, wn_lemmatizer.lemmatize))

    if debug: print([w[0] for w in text_analysis.top_words(texts, num_words=30)])

    print("Performing Porter Stemming")
    ## Porter stemming is more extreme than lemmatization and really makes different forms of words the same, but those root forms can be hard to read. 
    porter_stemmer = PorterStemmer()

    texts = texts.apply(lambda x: apply_fn_text(x, porter_stemmer.stem))

    if debug: print([w[0] for w in text_analysis.top_words(texts, num_words=30)])

    return texts


