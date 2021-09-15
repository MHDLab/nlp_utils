"""
functions for generating citation trees/graphs etc. 
"""

from nlp_utils.io import load_df_semantic
import pandas as pd
import networkx as nx
import numpy as np


def get_citations(con, df):

    df_both = pd.DataFrame(columns=df.columns)

    for id,row in df.iterrows():
        df_in = load_df_semantic(con, row['inCitations'])
        df_out = load_df_semantic(con, row['outCitations'])
        df_temp = pd.concat([df_in, df_out])
        df_both = pd.concat([df_both, df_temp])

    df_both = df_both.loc[~df_both.index.duplicated()]

    return df_both

def gen_citation_graph(df):
    G = nx.Graph()
    G.add_nodes_from(df.index.values)

    for idx, row in df.iterrows():
        cits = row['inCitations']
        for cit in cits:
            G.add_edge(idx, cit)

        cits = row['outCitations']
        for cit in cits:
            G.add_edge(idx, cit)

    return G

def trim_graph(G, con, frac_keep_factor =1, n_trim = 10000, min_connect=0):
    """
    removes nodes with fewer than a given number of edges
    """

    s_degrees = pd.Series(dict(G.degree()))

    ## Old method
    # num_edges = int(np.log(len(s_degrees)))/min_edge_factor
    # print('discading nodes with fewer than ' + str(num_edges) + ' edges')

    #First downselect to a given most connected. Speeds up looking through database to get citation counts 
    s_degrees = s_degrees.sort_values(ascending=False)[0:n_trim]
    
    s_degrees = s_degrees.where(s_degrees>min_connect).dropna()

    #only keep papers in database adn get the fraction of edges/(total in and out citations) for each publication
    df_all = load_df_semantic(con, s_degrees.index)
    n_citations = df_all['inCitations'].apply(len) + df_all['outCitations'].apply(len) 
    s_degrees = s_degrees.loc[n_citations.index] # Some edges are not in database
    frac_connected = s_degrees/n_citations
    
    avg_connected_frac = frac_connected.mean()
    print("Size after database checking: {}".format(len(s_degrees)))
    print("Average connected fraction: {:.3f}".format(avg_connected_frac))

    frac_keep = avg_connected_frac*frac_keep_factor
    print('discading nodes with fewer than {:.3f} fraction connected edges'.format(frac_keep))

    s_degrees = s_degrees.where(frac_connected > frac_keep).dropna()
    print("After trimming edges: " + str(len(s_degrees)))
 

    return G.subgraph(s_degrees.index)

def build_citation_community(df, con, n_iter=1,frac_keep_factor=1, n_trim=20000):

    df_temp = df
    for i in range(n_iter):
        print('---Step {}---'.format(i))
        print("Before graph generation: " +str(len(df_temp)))

        G = gen_citation_graph(df_temp)
        print("After Graph Generation: " + str(len(G.nodes)))

        if i > 0:
            G = trim_graph(G, con, frac_keep_factor, n_trim)

        df_2 = load_df_semantic(con, list(G.nodes))
        df_temp = pd.concat([df_temp, df_2])
        df_temp = df_temp.loc[~df_temp.index.duplicated()]

    return df_temp

if __name__ == '__main__':
    import sqlite3

    db_path = r'C:\Users\aspit\Git\MLEF-Energy-Storage\semantic_opencorpus\data\soc.db'
    con = sqlite3.connect(db_path)

    test_ids = [
        'fe09b0eee943efef3cdc3ec14db772fc400c6c08',
    ]

    df = load_df_semantic(con, test_ids)

    build_citation_community(df, con, n_iter=10, frac_keep_factor=0.5, n_trim=20000)






