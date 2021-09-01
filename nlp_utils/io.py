import sqlite3
import os
import pandas as pd



def load_df_semantic(con, ids, dataset='soc', cust_idx_name=None):
    """
    load a series of papers from either semantic open corpus(soc) or s2orc datasets.
    cust_idx_name: overrides default index to search through (e.g. doi)

    TODO: Perhaps names should be made consistent here. 
    """
    if dataset == 'soc':
        idx = 'id'
    elif dataset == 's2orc':
        idx = 'paper_id'
    else:
        raise ValueError("dataset must be 'soc' or 's2orc'")

    #search through database using a different index like doi
    if cust_idx_name != None:
        search_idx = cust_idx_name
    else:
        search_idx = idx


    table_info = con.execute(f'PRAGMA table_info(raw_text);').fetchall()
    columns = [t[1] for t in table_info]

    cursor = con.cursor()
    cursor.execute(
        "SELECT * from raw_text where {} in (\"{}\")".format(search_idx, "\", \"".join(ids))
        )
    results = cursor.fetchall()

    df = pd.DataFrame(results, columns=columns).set_index(idx)
    df['year'] = df['year'].astype(float).astype(int)

    df['years_ago'] = abs(2022 - df['year'])

    if dataset == 'soc':
        df['inCitations'] = df['inCitations'].apply(lambda x: x.strip('[]').replace("'", "").replace(" ", "").split(','))
        df['outCitations'] = df['outCitations'].apply(lambda x: x.strip('[]').replace("'", "").replace(" ", "").split(','))
    elif dataset == 's2orc':
        df['inbound_citations'] = df['inbound_citations'].apply(lambda x: x.strip('[]').split(','))
        df['outbound_citations'] = df['outbound_citations'].apply(lambda x: x.strip('[]').split(','))

    return df

def get_column_as_list(con, col_name, table_name):
    """
    returns a column from the database parsing as a list
    """

    ids = con.execute("select \"{}\" from {}".format(col_name, table_name)).fetchall()
    ids = [i[0] for i in ids if i[0] != None]

    return ids

def load_df_MA(db_path):
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    df_meta = pd.read_sql_query("SELECT * FROM raw_text", con, index_col='ID')
    df = pd.read_sql_query("SELECT * FROM processed_text", con, index_col='ID')

    if 'language' in df.columns:
        df = df[df['language'] == 'en'].dropna(subset=['processed_text'])
    
    df_out = pd.concat([df, df_meta.loc[df.index]], axis=1)

    df_out = df_out.dropna(subset=['processed_text'])

    return df_out 

if __name__ == '__main__':
    data_folder = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\data'

    df = load_df(os.path.join(data_folder, 'nlp_justenergystorage_100.db'))
