
def print_topic_words(topic_model, n_words=10):

    topics = topic_model.get_topics()
    for topic_n,topic in enumerate(topics):
        words,mis, sign  = zip(*topic)
        topic_str = str(topic_n+1)+': '+','.join(words[0:n_words])
        print(topic_str)


def print_top_docs(topic_model, titles):

    top_docs = topic_model.get_top_docs()
    for topic_n, topic_docs in enumerate(top_docs):
        docs,probs = zip(*topic_docs)
        docs = [str(titles[d])[0:100] for d in docs]
        topic_str = str(topic_n+1)+': \n'+',\n'.join(docs)
        print(topic_str)