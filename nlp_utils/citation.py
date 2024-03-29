"""
functions for generating citation trees/graphs etc. 
"""

from .fileio import load_df_semantic
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


from .fileio import load_df_semantic

def gen_citation_tree(G, con, cit_field, add_new=True):
    """
    adds edges for each citation in cit_field (inCitations, outCitations)
    add_new: add edges for citations not existing in original graph
    remove_nodes_not_in_db: after growing citation graph, remove nodes that don't exist in database.
    """

    current_ids = list(G.nodes())
    df = load_df_semantic(con, current_ids)

    #graph could include ids not in database, so iterate through df (those found in db)
    for idx, row in df.iterrows():

        if cit_field == 'both':
            cits = row['inCitations']
            cits.extend(row['outCitations'])
        else:
            cits = row[cit_field]

        if not add_new:
            cits = [cit for cit in cits if cit in current_ids]

        for cit in cits:
            G.add_edge(idx, cit)

    return G

def trim_graph_num_edges(G, min_edges):
    """
    trims a graph, removing nodes with fewer than min_edges
    """

    print("removing all papers with less than {} edges".format(min_edges))
    s_degrees = pd.Series(dict(G.degree()))
    s_degrees = s_degrees.where(s_degrees>= min_edges).dropna()
    G = G.subgraph(s_degrees.index)
    print("after removing min connection: {}".format(len(G.nodes())))

    G = nx.Graph(G)
    return G

def trim_graph_size(G, max_size):
    """
    trims the graph down to a maximum size. papers with low numbers of degrees are dropped until the maximum number is reached. 
    """
    s_degrees = pd.Series(dict(G.degree()))


    val_counts = s_degrees.value_counts().sort_index(ascending=False)
    edges_cutoff = val_counts.where(val_counts.cumsum()<max_size).dropna().astype(int).index[-1]

    s_degrees = s_degrees.where(s_degrees>edges_cutoff).dropna()
    # s_degrees.hist(bins=100)

    G = G.subgraph(s_degrees.index)
    print("removing all papers with less than {} edges to reduce graph to size {}".format(edges_cutoff, len(G.nodes)))
    G = nx.Graph(G)
    return G

def get_citation_info(G, con, drop_nonexist = True):
    #get the fraction of edges/(total in and out citations) for each publication
    df_all = load_df_semantic(con, list(G.nodes))
    inCitations = df_all['inCitations'].apply(len)
    outCitations = df_all['outCitations'].apply(len) 
    years_ago = df_all['years_ago']

    #only keep papers in database 
    if drop_nonexist:
        G = G.subgraph(df_all.index)

    #perhaps theres a more efficient way of doing this
    for node in G.nodes:
        G.nodes[node]['inCitations'] = inCitations[node]
        G.nodes[node]['outCitations'] = outCitations[node]
        G.nodes[node]['frac_connected'] = G.degree()[node]/(G.nodes[node]['inCitations'] + G.nodes[node]['outCitations'] )
        G.nodes[node]['inCitsPerYear'] = inCitations[node]/years_ago[node]
    return G


def trim_graph_fraction(G, max_size):
    """
    removes nodes with fewer than the average*frac_keep_factor fraction of connected edges
    get_frac_connected must be run first... TODO: this should problably just be integrated into one funciton. 
    """

    frac_connected = pd.Series(nx.get_node_attributes(G, 'frac_connected')).sort_values(ascending=False)

    keep = frac_connected.iloc[0:max_size]
    min_fraction = keep.iloc[-1]
 
    G = G.subgraph(keep.index)
    G = nx.Graph(G)

    print('discading nodes with fewer than {:.3f} fraction connected edges to reduce graph to size {}'.format(min_fraction, len(G.nodes)))
    return G

# TODO: should be able to have one function to dowselect arbitrary size basedon attr
def trim_graph_inCits(G, max_size):

    frac_connected = pd.Series(nx.get_node_attributes(G, 'inCitations')).sort_values(ascending=False)

    keep = frac_connected.iloc[0:max_size]
    min_fraction = keep.iloc[-1]
    print('discading nodes with fewer than {:.3f} in citations per year'.format(min_fraction))
 
    G = G.subgraph(keep.index)
    G = nx.Graph(G)

    return G

def trim_connected_components(G, num_cc):
    # Find connected components (islands) and keep the largest num_cc
    cc= pd.Series(nx.connected_components(G))
    cc_len = cc.apply(len).sort_values(ascending=False)
    cc_keep = cc.loc[cc_len.iloc[0:num_cc].index]
    ids_keep = set(set().union(*cc_keep.values))

    G = G.subgraph(ids_keep)
    G = nx.Graph(G)
    return G

##Obsolete, need to remake?
# def build_citation_community(df, con, n_iter=1,frac_keep_factor=1, n_trim=20000):

#     df_temp = df
#     for i in range(n_iter):
#         print('---Step {}---'.format(i))
#         print("Before graph generation: " +str(len(df_temp)))

#         G = gen_citation_graph(df_temp)
#         print("After Graph Generation: " + str(len(G.nodes)))

#         if i > 0:
#             G = trim_graph(G, con, frac_keep_factor, n_trim)

#         df_2 = load_df_semantic(con, list(G.nodes))
#         df_temp = pd.concat([df_temp, df_2])
#         df_temp = df_temp.loc[~df_temp.index.duplicated()]

#     return df_temp

if __name__ == '__main__':
    import sqlite3

    db_path = r'C:\Users\aspit\Git\MLEF-Energy-Storage\semantic_opencorpus\data\soc.db'
    con = sqlite3.connect(db_path)

    test_ids = [
        'fe09b0eee943efef3cdc3ec14db772fc400c6c08',
    ]

    df = load_df_semantic(con, test_ids)

    build_citation_community(df, con, n_iter=10, frac_keep_factor=0.5, n_trim=20000)






