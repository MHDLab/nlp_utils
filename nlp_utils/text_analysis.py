from collections import defaultdict

def gen_wordcount_dict(corpus, word_filter):
    dic = defaultdict(int)
    for text in corpus:
        for word in text:
            if word_filter is None:
                dic[word] = dic[word] + 1
            else:
                if word in word_filter:
                    dic[word] = dic[word] + 1    
    return dic

def top_words(corpus, num_words = 10, word_filter = None):
    dic = gen_wordcount_dict(corpus, word_filter)
    top = sorted(dic.items(), key= lambda x:x[1], reverse=True)[0:num_words]

    return top 