
import pandas as pd

def calc_topics_year(df_doc_topic_probs, s_year, norm_each_topic=True):
    """
    Calculate the topic probability trends (how the proability of a topic is changing over the years). 
    A dataframe with index = year and column = topic is created that represents the
    trend of each topic throughout the literature.

    df_doc_topic_probs: document topic probability matrix, index: doc id, column: topic name
    s_year: pandas series with index of document id and value of year
    """

    years = set(s_year.values)

    df_topicsyear = pd.DataFrame(index=s_year.index, columns=df_doc_topic_probs.columns, dtype=float)
    df_topicsyear.index.name = 'year'

    for year in set(s_year.values):
        ids = s_year[s_year == year].index.values

        #sum up the topic probability over all articles and normalize, giving a a probability distribution for the most likely topics of that year. 
        topics_year = df_doc_topic_probs.loc[ids].sum()
        topics_year = topics_year/topics_year.sum()

        #Insert into the topics trend dataframe. 
        df_topicsyear.loc[year] = topics_year

    #TODO: remove this
    df_topicsyear = df_topicsyear.dropna() #SOC dataset has some missing values for some reason. 


    if norm_each_topic:
        #normailze each topic by it's cumulative probability over all years. Some topics
        #have high absolute slopes just because they are generally more probable, and
        #this is a way to account for that such that a smaller overall topic that is
        #growing in popularity will be noticed. This will remove the normalization over
        #a given year. Also, perhaps a normalized slope could instead just be calculated
        #after the fitting process, to be used in ranking without having to alter the
        #data in this way.  
        for topic_id in df_topicsyear:
            df_topicsyear[topic_id] = df_topicsyear[topic_id]/sum(df_topicsyear[topic_id])

    return df_topicsyear

