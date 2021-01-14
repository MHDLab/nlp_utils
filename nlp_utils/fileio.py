import sqlite3
import pandas as pd

def load_df(db_path):
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    df = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID')
    df = df.dropna(subset=['processed_text'])
    df = df[df['language'] == 'en']

    return df 