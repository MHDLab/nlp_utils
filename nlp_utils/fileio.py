import sqlite3
import os
import pandas as pd
import re

#https://stackoverflow.com/questions/5365451/problem-with-regexp-python-and-sqlite
def sqlite_regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

def gen_ids_regex(con, regex, idx_name, search_fields, search_limit, output_limit):
    """
    For each regular expression search if it exists in any of the search_fields
    """

    con.create_function("REGEXP", 2, sqlite_regexp)

    cursor = con.cursor()

    # put -- to comment out line
    execute_str = """
        SELECT {} FROM
        --raw_text
        (SELECT * FROM raw_text LIMIT {})
        WHERE {} REGEXP '{}'
    """.format(idx_name, search_limit, search_fields[0], regex)

    for search_field in search_fields[1:]:
        execute_str = execute_str + " OR {} REGEXP '{}'".format(search_field, regex)

    execute_str = execute_str + ' LIMIT {}'.format(output_limit)

    print('Excecuting Query:')
    print(execute_str)

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
    """

    #search through database using a different index like doi
    if cust_idx_name != None:
        search_idx = cust_idx_name
    else:
        search_idx = 'id'


    table_info = con.execute(f'PRAGMA table_info(raw_text);').fetchall()
    columns = [t[1] for t in table_info]

    cursor = con.cursor()
    results = []
    for id in ids:
        cursor.execute(
            "SELECT * from raw_text where {} is \"{}\"".format('id', id)
            )
        results.extend(cursor.fetchall())

    df = pd.DataFrame(results, columns=columns).set_index(search_idx)
    df['year'] = df['year'].astype(float).astype(int)

    df['years_ago'] = abs(2020 - df['year']) #TODO: how to find 'max year' of dataset 

    if dataset == 'soc':
        df['inCitations'] = df['inCitations'].apply(lambda x: x.strip('[]').replace("'", "").replace(" ", "").split(','))
        df['outCitations'] = df['outCitations'].apply(lambda x: x.strip('[]').replace("'", "").replace(" ", "").split(','))
        df = df.rename({'s2Url': 'display_url'}, axis=1)
    elif dataset == 's2orc':
        df['inbound_citations'] = df['inbound_citations'].apply(lambda x: x.strip('[]').split(','))
        df['outbound_citations'] = df['outbound_citations'].apply(lambda x: x.strip('[]').split(','))
        df = df.rename({'s2_url': 'display_url'}, axis=1)

    df['inCits_per_year'] = df['inbound_citations'].apply(len)
    return df

def get_columns_as_df(con, columns, search_limit=None, dataset='soc', table_name='raw_text'):
    """
    returns a column from the database parsing as a list
    """

    column_str = ", ".join(['id', *columns])

    query  = "SELECT {} FROM {}".format(column_str, table_name)

    if search_limit != None:
        query += " LIMIT {}".format(search_limit)

    df = pd.read_sql_query(query ,con, index_col='id')

    return df


def load_df_SEAMs(con):
    """
    Loads in the SEAMS data in a format that is consistent with SOC data The
    seams data consists of processed OCR text data in a sqlite database. There
    is a metadata csv file that is stored with it on the sharepoint. This
    metadata file has the same information as the metadata table in the database
    but also includes the edx pdf url. 
    """
    df = pd.read_sql_query("SELECT * FROM texts", con, index_col='id')
    df_meta = pd.read_sql_query("SELECT * FROM metadata", con, index_col='id')
    df = pd.concat([df, df_meta], axis=1)

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
    ids = gen_ids_regex(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e4))

    df = load_df_semantic(con, ids)
    print(df.head())