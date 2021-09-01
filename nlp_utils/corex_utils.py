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