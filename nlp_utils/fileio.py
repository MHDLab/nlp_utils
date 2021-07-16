import sqlite3
import os
import pandas as pd

def load_df(db_path):
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
