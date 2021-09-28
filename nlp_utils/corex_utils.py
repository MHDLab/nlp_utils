import numpy as np
import pandas as pd


def get_s_topic_words(topic_model, n_words = 6):
    """
    CorEx also returns the "sign" of each word, which is either 1 or -1. If the sign is -1, then that means the absence of a word is informative in that topic, rather than its presence.
    """

    topic_names = ['topic_' + str(i) for i in range(topic_model.n_hidden)]
    s_topic_words = pd.Series(index=topic_names, dtype=str)

    topics = topic_model.get_topics(n_words)
    for i,topic in enumerate(topics):
        topic_name = 'topic_' + str(i)
        outstr = ''
        for word, mis, sign in topic:
            sign = '+' if sign == 1.0 else '-'
            outstr = outstr + sign + word + ' '

        s_topic_words[topic_name] = outstr

    return s_topic_words

#TODO: output data vs printing like get_s_topic_words
def print_top_docs(topic_model, titles):

    top_docs = topic_model.get_top_docs()
    for topic_n, topic_docs in enumerate(top_docs):
        docs,probs = zip(*topic_docs)
        docs = [str(titles[d])[0:100] for d in docs]
        topic_str = str(topic_n+1)+': \n'+',\n'.join(docs)
        print(topic_str)


def anchors_to_fixed_bigrams(anchor_words):
    fixed_bigrams = []

    for word in anchor_words:
        if type(word) == list:
                fixed_bigrams.extend(word)
        else:
            fixed_bigrams.append(word)

    fixed_bigrams = [w for w in fixed_bigrams if '_' in w]

    return fixed_bigrams 

from .gensim_utils import calc_cov
import xarray as xr

def calc_cov_corex(topic_model, topic_names, doc_names):
    """
    calculate the covariance matrix (topic coocurence) used to define the edges in the graph
    see here:  https://www.aclweb.org/anthology/W14-3112.pdf
    """

    doc_topic_prob = topic_model.p_y_given_x

    n_topics = doc_topic_prob.shape[1]
    n_docs = doc_topic_prob.shape[0]
    
    da_doc_topic = xr.DataArray(doc_topic_prob, coords= {'topic': topic_names, 'doc' : doc_names}, dims = ['doc', 'topic'])

    #Normalize so each topic has total probability one (what does this do in combination with below?)
    theta_ij = da_doc_topic/da_doc_topic.sum('doc')

    #Then normalize so each document has total probability 1
    gamma_di = theta_ij/theta_ij.sum('topic')

    gamma_i = (1/n_docs)*gamma_di.sum('doc')
    gamma_di_sub = gamma_di - gamma_i

    sigma = calc_cov(gamma_di_sub.values)

    da_sigma = xr.DataArray(sigma, coords = {'topic_i': topic_names, 'topic_j': topic_names}, dims = ['topic_i', 'topic_j'])

    return da_sigma, da_doc_topic


def corex_edge_info(doc_topic_probs):
    """
    Extracts the top papers for each set of two topics of a corex model. words belong to only one topic so it's not clear how to get edge keywords yet, so a dummy keyword dataframe is returned for now. 
    """
    num_docs = doc_topic_probs.shape[0]
    num_topics = doc_topic_probs.shape[1]
    
    #create new matrixes for the 3D data
    # edge_term = np.ndarray((num_topics, num_topics, num_words))
    doc_edge_probs = np.ndarray((num_docs, num_topics, num_topics))
    row_labels = []

    #fill the matrices and then reduced to 2D
    for t1 in range(num_topics):
        for t2 in range(num_topics):
            row_labels.append('topic_'+str(t1)+ ', topic_'+str(t2))
            if t1 == t2:
                doc_edge_probs[:, t1, t2] = np.zeros((num_docs))
            else:
                doc_edge_probs[:, t1, t2] = doc_topic_probs[:,t1] * doc_topic_probs[:,t2]
    doc_edge_probs = np.reshape(doc_edge_probs, (num_docs, num_topics**2))

    
    #Make a dummy edge keyword. Kept method of making dummy for all ranks for reference. 
    # TODO: figure out a way to get edge keywords or clean up below 
    df_edgekeywords = pd.DataFrame(index=row_labels)
    for i in range(1,11):
        df_edgekeywords['rank_{}'.format(i)] = [" "]*len(row_labels)
    df_edgekeywords.index.name = 'topic'

    df_edgekeywords = df_edgekeywords[['rank_1']]
    df_edgekeywords['rank_1'] = "Edge keywords are not available with CorEx topic model."

    df_doc_edge_probs = pd.DataFrame(doc_edge_probs, columns=df_edgekeywords.index, )
    df_doc_edge_probs = df_doc_edge_probs.div(df_doc_edge_probs.sum(axis=1), axis=0) # Normalize each edge...

    return df_edgekeywords, df_doc_edge_probs 

