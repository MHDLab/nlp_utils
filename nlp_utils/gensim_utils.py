import gensim

import pandas as pd
import numpy as np

from tmtoolkit.topicmod.model_io import ldamodel_top_topic_words
from tmtoolkit.topicmod.model_stats import topic_word_relevance
from tmtoolkit.bow.bow_stats import doc_lengths


def gensim_bigram(texts, bigram_kwargs):
    bigram = gensim.models.Phrases(texts, **bigram_kwargs)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    texts_bigram = [bigram_mod[doc] for doc in texts]

    return texts_bigram

def basic_gensim_lda(texts, lda_kwargs):
    id2word = gensim.corpora.Dictionary(texts)
    data_words = [id2word.doc2bow(doc) for doc in texts]

    lda_model = gensim.models.LdaModel(
                                    data_words,
                                    **lda_kwargs,
                                    random_state=42
    )

    return id2word, data_words, lda_model


def gensim_lda_bigram(texts, bigram_kwargs, lda_kwargs):
    """
    Basic model generation pipeline to generate bigrams from a series of texts and then generate an LDA model. 

    bigram_kwargs and lda_kwargs are dictionaries for the bigram phraser and LDA model generation. 
    """

    texts_bigram = gensim_bigram(texts, bigram_kwargs)
    id2word, data_words, lda_model = basic_gensim_lda(texts, lda_kwargs)
    

    return texts_bigram, id2word, data_words, lda_model


def get_vocab_docl(data_words, id2word):
    """
    Extract a vocabulary list and length of each document
    """
    dtm = gensim.matutils.corpus2csc(data_words).astype(int).T
    doc_l = doc_lengths(dtm)
    vocab = np.array([id2word[i] for i in range(dtm.shape[1])])

    return vocab, doc_l

def gensim_topic_info(lda_model, data_words, id2word, lambda_=0.6):
    """
    Extracts the top keywords for each topic, and the document-topic probability matrix from an LDA model
    """
    vocab, doc_l = get_vocab_docl(data_words, id2word)

    doc_topic_probs = list(lda_model.get_document_topics(data_words, minimum_probability=0))
    doc_topic_probs = gensim.matutils.corpus2csc(doc_topic_probs).T.toarray()

    topic_term = lda_model.get_topics()
    topic_word_rel = topic_word_relevance(topic_term, doc_topic_probs, doc_l, lambda_=lambda_)
    df_topickeywords = ldamodel_top_topic_words(topic_word_rel, vocab, top_n=10, val_fmt=r"{lbl}")

    return df_topickeywords, doc_topic_probs

def gensim_edge_info(lda_model, data_words, id2word, doc_topic_probs, lambda_=0.6):
    """
    Extracts the top keywords for each set of two topics, and the document-topic probability matrix from an LDA model
    """
    vocab, doc_l = get_vocab_docl(data_words, id2word)


    topic_term = lda_model.get_topics() #the probability for each word in each topic, shape(num_topics, vocabulary_size) = (20, 55132)
    
    num_docs = doc_topic_probs.shape[0]
    num_topics = topic_term.shape[0]
    num_words = topic_term.shape[1]
    
    #create new matrixes for the 3D data
    edge_term = np.ndarray((num_topics, num_topics, num_words))
    doc_edge_probs = np.ndarray((num_docs, num_topics, num_topics))
    row_labels = []

    #fill the matrices and then reduced to 2D
    for t1 in range(num_topics):
        for t2 in range(num_topics):
            edge_term[t1,t2,:] = topic_term[t1,:] * topic_term[t2,:]
            row_labels.append('topic_'+str(t1+1)+ ', topic_'+str(t2+1))
            if t1 == t2:
                doc_edge_probs[:, t1, t2] = np.zeros((num_docs))
            else:
                doc_edge_probs[:, t1, t2] = doc_topic_probs[:,t1] * doc_topic_probs[:,t2]
    edge_term = np.reshape(edge_term, (num_topics**2, num_words)) #(400, 55132)
    doc_edge_probs = np.reshape(doc_edge_probs, (num_docs, num_topics**2))

    edge_word_rel = topic_word_relevance(edge_term, doc_edge_probs, doc_l, lambda_=lambda_) #identify the most relevant word accounting for marginal probabilities, shape (num_topics, vocabulary_size)
    df_edgekeywords = ldamodel_top_topic_words(edge_word_rel, vocab, top_n=10, row_labels=row_labels, val_fmt=r"{lbl}")

    return df_edgekeywords, doc_edge_probs

def bigram_stats(data_words, id2word):
    vocab, docl = get_vocab_docl(data_words, id2word)

    bigram_count = [1 if '_' in c else 0 for c in vocab ]
    num_bigrams = sum(bigram_count)
    total_words = len(vocab)

    return num_bigrams, total_words


#Topic covariance 
#https://www.aclweb.org/anthology/W14-3112.pdf

from numba import jit
import xarray as xr

@jit(nopython=True)
def calc_cov(gamma_di_sub):
    """
    Calculate the covariance matrix sigma(i,j) from gamma_di_sub = gamma_di - gamma_i
    See the link above. 
    This functuion only uses numpy arrays and can be sped up with numba, but this doesn't seem to work on heroku. 
    """
    n_docs = gamma_di_sub.shape[0]
    n_topics = gamma_di_sub.shape[1]

    sigma = np.zeros((n_topics,n_topics))
    sigma
    for i in range(n_topics):
        for j in range(n_topics):
            sum = 0
            for doc in range(n_docs):
                sum = sum + gamma_di_sub[doc][i]*gamma_di_sub[doc][j]
            sigma[i][j] = sum

    return sigma

def calc_cov_wrap(df_doc_topic_probs, topic_names):
    """
    Calculates the topic covariance matrix from the document topic probability matrix (theta_di)

    First calculate gamma_di - gamma_i which is the main factor in covariance matrix
    """
    
    #

    #All basically 1...Think I already normalized
    gamma_di = df_doc_topic_probs#.values

    gamma_i = (1/len(df_doc_topic_probs.index))*df_doc_topic_probs.sum()

    gamma_di_sub = gamma_di.copy()

    for topic in gamma_i.index:
        gamma_di_sub[topic] = gamma_di[topic] - gamma_i[topic]

    gamma_di_sub = gamma_di_sub.values

    #Then run numba optimized function
    sigma = calc_cov(gamma_di_sub)

    da_sigma = xr.DataArray(sigma, coords = {'topic_i': topic_names, 'topic_j': topic_names}, dims = ['topic_i', 'topic_j'])

    return da_sigma