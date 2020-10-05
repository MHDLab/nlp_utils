#visualization imports
import numpy as np
import pandas as pd
import os

from sklearn.decomposition import LatentDirichletAllocation

pd.options.mode.chained_assignment = None

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorization(df, k):
    """
    Vectorization for topic modeling starts here.
    """
    vectorizers = []

    for ii in range(0, k):
        # Creating a vectorizer; removed stop_words = 'english'
        vectorizers.append(CountVectorizer(ngram_range = (1,2), min_df = 2, max_df = 0.80, lowercase = True, token_pattern = '[a-zA-Z\-][a-zA-Z\-]{2,}'))

    vectorized_data = []

    for current_cluster, cvec in enumerate(vectorizers):
        vectorized_data.append(cvec.fit_transform(df.loc[df['y_pred'] == current_cluster, 'processed_text']))

    # print(len(vectorized_data))
    return vectorizers, vectorized_data

def selected_topics(model, vectorizer, top_n):
    """
    Function called by topic_modelling to find keywords for topics.
    """
    current_words = []
    keywords = []

    for idx, topic in enumerate(model.components_):
        #topic.argsort gives indicies of of top_n words associated with topic 
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])

    keywords.sort(key = lambda x: x[1])
    keywords.reverse()
    return_values = []

    for ii in keywords:
        return_values.append(ii[0])

    return return_values

def topic_modelling(k, vectorizers, vectorized_data, NUM_TOPICS_PER_CLUSTER = 20, top_n = 3):
    """
    Topic modelling starts using LDA machine-learning model starts here.

    NUM_TOPICS_PER_CLUSTER number of topics for lda model of each cluster 

    top_n:how many words are added for each topic to the total cluster topic
    list. The top_n words from a previous topic will not be included, so the
    total number of words will be somewhat less that
    NUM_TOPICS_PER_CLUSTER*top_n
    """
    

    lda_models = []
    for ii in range(0, k):
        # Latent Dirichlet Allocation Model
        lda = LatentDirichletAllocation(n_components = NUM_TOPICS_PER_CLUSTER, max_iter = 10, learning_method = 'online', verbose = False, random_state = 42)
        lda_models.append(lda)

    clusters_lda_data = []

    for current_cluster, lda in enumerate(lda_models):
        # print("Current Cluster: " + str(current_cluster))
        if vectorized_data[current_cluster] != None:
            clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))

    all_keywords = []

    for current_vectorizer, lda in enumerate(lda_models):
        # print("Current Cluster: " + str(current_vectorizer))
        if vectorized_data[current_vectorizer] != None:
            all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer], top_n))

    return all_keywords


def find_keywords(corpus, N):
    """Find top words of a given text, probably only useful for long texts"""
    corpus = [corpus]
    vectorizer = TfidfVectorizer(ngram_range = (1,1))
    vectors = vectorizer.fit_transform(corpus)
    names = vectorizer.get_feature_names()
    data = vectors.todense().tolist()

    # Create a dataframe with the results
    s = pd.Series(data[0], index=names)

    keywords = s.sort_values(ascending=False).iloc[0:10].index.values


    keywords = ','.join(str(v) for v in keywords)

    return keywords