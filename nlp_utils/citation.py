"""
functions for generating citation trees/graphs etc. 
"""

from nlp_utils.io import load_df_semantic
import pandas as pd
import networkx as nx
import numpy as np
import pandas as pd


def get_citations(con, df):

    df_both = pd.DataFrame(columns=df.columns)

    for id,row in df.iterrows():
        df_in = load_df_semantic(con, row['inCitations'])
        df_out = load_df_semantic(con, row['outCitations'])
        df_temp = pd.concat([df_in, df_out])
        df_both = pd.concat([df_both, df_temp])

    df_both = df_both.loc[~df_both.index.duplicated()]

    return df_both

def gen_citation_tree(G, df):

    for idx, row in df.iterrows():
        cits = row['inCitations']
        for cit in cits:
            G.add_edge(idx, cit)

        cits = row['outCitations']
        for cit in cits:
            G.add_edge(idx, cit)
    
    return G

def get_frac_connected(G, con, drop_nonexist = True):
    #get the fraction of edges/(total in and out citations) for each publication
    df_all = load_df_semantic(con, list(G.nodes))
    n_citations = df_all['inCitations'].apply(len) + df_all['outCitations'].apply(len) 

    #only keep papers in database 
    if drop_nonexist:
        G = G.subgraph(n_citations.index)

    #perhaps theres a more efficient way of doing this
    for node in G.nodes:
        G.nodes[node]['total_cits'] = n_citations[node]
        G.nodes[node]['frac_connected'] = G.degree()[node]/n_citations[node]
        
    return G


def trim_graph(G, frac_keep_factor =1, n_trim = 10000, min_connect=0):
    """
    removes nodes with fewer than a given number of edges
    """
    
    frac_connected = pd.Series(nx.get_node_attributes(G, 'frac_connected'))

    avg_connected_frac = frac_connected.mean()
    print("Size after database checking: {}".format(len(frac_connected)))
    print("Average connected fraction: {:.3f}".format(avg_connected_frac))

    frac_keep = avg_connected_frac*frac_keep_factor
    print('discading nodes with fewer than {:.3f} fraction connected edges'.format(frac_keep))

    frac_connected = frac_connected.where(frac_connected > frac_keep).dropna()
    print("After trimming edges: " + str(len(frac_connected)))
 
    G = G.subgraph(frac_connected.index)
    G = nx.Graph(G)

    s_degrees = pd.Series(dict(G.degree()))
    s_degrees = s_degrees.where(s_degrees>min_connect).dropna()
    G = G.subgraph(s_degrees.index)

    print("After removing min count: " + str(len(s_degrees)))

    return G

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






