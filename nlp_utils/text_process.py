"""
https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py
Seems like gensim convention is 

documents: single string for each document
texts: list of string words for each document
corpus: digitized bag of words for
"""

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from mat2vec.processing import MaterialsTextProcessor

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

def stopword_removal(text, stopwords):
    text_out = []
    for word in text:   
        if word.lower() not in stopwords:
            text_out.append(word)
    
    return text_out

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, post_stopwords = None):
        self.stopwords = stopwords.words('english')
        self.post_stopwords = post_stopwords #remove at the end

        self.materials_text_processor = MaterialsTextProcessor()

        # self.wn_lemmatizer = WordNetLemmatizer()
        self.porter_stemmer = PorterStemmer()

    def fit(self, texts, y=None):
        return self
    
    def transform(self, texts):
        texts_out = [] #TODO: Can't chain in pipeline using yield...
        for text in texts:
            text = self.normalize(text)

            text = self.materials_text_processor.process(text, exclude_punct=True, replace_elements = True)[0]
            text = [t for t in text if t != '<nUm>']

            #Porter stemmer removes capitalization...
            text = [self.porter_stemmer.stem(t) for t in text]

            if self.post_stopwords is not None:
                text = [t for t in text if t not in self.post_stopwords]

            texts_out.append(text)
        return texts_out


    def normalize(self, text):
        translator = str.maketrans('','',string.punctuation.replace('-',''))
        text_out = []

        for word in text:
            if word.lower() not in self.stopwords:
                word = word.translate(translator)
                word = word.replace(b'\xe2\x80\x93'.decode('utf-8'), '_').replace('-','_')
                if len(word): #TODO: figure out how to remove in one step
                    text_out.append(word)
        
        return text_out



if __name__ == '__main__':

    import sqlite3
    import os
    from fileio import load_df_semantic

    DATASET_DIR = r'E:'
    db_path = os.path.join(DATASET_DIR, 'soc.db')

    #Get IDS
    id = '767fa9da3adf2aed0acdaea699ac8783c7859b0f'
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    df = load_df_semantic(con, [id])      
    docs = df['title'] + ' ' + df['paperAbstract']
    texts = docs.apply(str.split)


    print("Before")
    print(texts.values)
    textnorm = TextNormalizer()
    print("After")
    print(list(textnorm.transform(texts)))