import sqlite3
import os
import pandas as pd

def gen_ids_searchterm(con, regex, idx_name, search_fields, search_limit, output_limit):
    """
    For each regular expression search if it exists in any of the search_fields
    """
    cursor = con.cursor()

    # put -- to comment out line
    execute_str = """
        SELECT {} FROM
        --raw_text
        (SELECT * FROM raw_text LIMIT {})
        WHERE {} LIKE '{}'
    """.format(idx_name, search_limit, search_fields[0], regex)

    for search_field in search_fields[1:]:
        execute_str = execute_str + " OR {} like '{}'".format(search_field, regex)

    execute_str = execute_str + ' LIMIT {}'.format(output_limit)
    cursor.execute(execute_str)
    ## Just get count, seems to take as long. 
    # cursor.execute("SELECT COUNT(*) FROM raw_text WHERE abstract LIKE '%flywheel energy storage%'")

    results = cursor.fetchall()

    print("Num Results: " + str(len(results)))

    ids = [r[0] for r in results]

    return ids


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
        df = df.rename({'s2Url': 'display_url'}, axis=1)
    elif dataset == 's2orc':
        df['inbound_citations'] = df['inbound_citations'].apply(lambda x: x.strip('[]').split(','))
        df['outbound_citations'] = df['outbound_citations'].apply(lambda x: x.strip('[]').split(','))
        df = df.rename({'s2_url': 'display_url'}, axis=1)
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

def load_df_SEAMs(db_folder):
    """
    Loads in the SEAMS data in a format that is consistent with SOC data The
    seams data consists of processed OCR text data in a sqlite database. There
    is a metadata csv file that is stored with it on the sharepoint. This
    metadata file has the same information as the metadata table in the database
    but also includes the edx pdf url. 

    TODO: reorganize and simplify SEAMS data storage, only using OCR text in database. 

    this function takes in the folder path and loads both files into one dataframe
    """
    db_path = os.path.join(db_folder, 'seamsnlp_final.db')
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID')

    metadata_path = os.path.join(db_folder, 'SEAMs_metadata.csv')
    df_meta = pd.read_csv(metadata_path, index_col=0)
    # df_meta = pd.read_sql_query("SELECT * FROM metadata", con, index_col='ID')

    df = pd.concat([df, df_meta], axis=1)

    df.index.name = 'id'
    df = df.rename({
        'Title': 'title',
        'Year':'year',
        'pdf_url': 'display_url'
    }, axis=1)

    return df

if __name__ == '__main__':

    db_folder = r'E:\\'

    con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))

    regex = '%energy storage%'
    ids = gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e4))

    df = load_df_semantic(con, ids)
    print(df.head())