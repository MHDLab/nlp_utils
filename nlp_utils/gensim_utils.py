import gensim

import pandas as pd
import numpy as np

from tmtoolkit.topicmod.model_io import ldamodel_top_topic_words
from tmtoolkit.topicmod.model_stats import topic_word_relevance
from tmtoolkit.bow.bow_stats import doc_lengths




def gensim_lda_bigram(texts, bigram_kwargs, lda_kwargs):

    bigram = gensim.models.Phrases(texts, **bigram_kwargs)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    texts_bigram = [bigram_mod[doc] for doc in texts]

    id2word = gensim.corpora.Dictionary(texts_bigram)
    data_words = [id2word.doc2bow(doc) for doc in texts_bigram]

    lda_model = gensim.models.LdaModel(
                                    data_words,
                                    **lda_kwargs,
                                    random_state=42
    )

    return texts_bigram, id2word, data_words, lda_model


def get_vocab_docl(data_words, id2word):

    dtm = gensim.matutils.corpus2csc(data_words).astype(int).T
    doc_l = doc_lengths(dtm)
    vocab = np.array([id2word[i] for i in range(dtm.shape[1])])

    return vocab, doc_l

def gensim_topic_info(lda_model, data_words, id2word, lambda_=0.6):

    vocab, doc_l = get_vocab_docl(data_words, id2word)

    doc_topic_probs = list(lda_model.get_document_topics(data_words, minimum_probability=0))
    doc_topic_probs = gensim.matutils.corpus2csc(doc_topic_probs).T.toarray()

    topic_term = lda_model.get_topics()
    topic_word_rel = topic_word_relevance(topic_term, doc_topic_probs, doc_l, lambda_=lambda_)
    df_topickeywords = ldamodel_top_topic_words(topic_word_rel, vocab, top_n=10, val_fmt=r"{lbl}")

    return df_topickeywords, doc_topic_probs


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
    
    #First calculat gamma_di - gamma_i which is the main factor in covariance matrix

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