import csv
import pandas as pd
import sqlite3


def loadData(file):
    return pd.read_csv(file, header=0)

api_df_csv = loadData('lovoo_v3_users_api-results.csv')
object_df_csv = loadData('lovoo_v3_users_instances.csv')
csv_merged = pd.merge(api_df_csv, object_df_csv, on="userId") 
sorted_csv_merged = csv_merged.sort_index(inplace=True)

dbfile = 'users_lovoo_v3.db'
con = sqlite3.connect(dbfile)
cur = con.cursor()
object_df_table = pd.read_sql_query("SELECT * FROM objects", con)
api_df_table = pd.read_sql_query("SELECT * FROM api", con)
db_merged = pd.merge(object_df_table, api_df_table, on="userId") 
sorted_db_merged = db_merged.sort_index(inplace=True)
con.close()

print(sorted_csv_merged == sorted_db_merged)