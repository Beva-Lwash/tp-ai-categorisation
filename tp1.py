import csv
import pandas as pandas
import sqlite3
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import plotly
from ipydatagrid import DataGrid
import openpyxl
import numpy as np

pandas.set_option('display.max_rows', 10)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', 10)

'''
def loadData(file):
    return pandas.read_csv(file, header=0)

api_df_csv = loadData('lovoo_v3_users_api-results.csv')
object_df_csv = loadData('lovoo_v3_users_instances.csv')
csv_merged = pandas.merge(api_df_csv, object_df_csv, on="userId") 
sorted_csv_merged = csv_merged.sort_index(inplace=True)
'''


dbfile = 'users_lovoo_v3.db'
con = sqlite3.connect(dbfile)
cur = con.cursor()
object_df_table = pandas.read_sql_query("SELECT * FROM objects", con)
api_df_table = pandas.read_sql_query("SELECT * FROM api", con)
con.close()
db_merged = pandas.merge(object_df_table, api_df_table, on="userId") 

db_merged_normalized = db_merged.copy()
date_columns = ['lastOnline', 'lastOnlineDate']
integer_columns = ['age_x', 'counts_pictures_x', 'counts_profileVisits_x', 'counts_kisses_x', 'lastOnlineTs', 'lang_count_x', 'countDetails', 'distance_x', 'age_y', 'counts_details', 'counts_pictures_y', 'counts_profileVisits_y', 'counts_kisses_y', 'counts_fans', 'counts_g', 'distance_y', 'lang_count_y', 'lastOnlineTime']

db_merged_normalized['lastOnlineTs'].replace('', np.nan, inplace=True)
db_merged_normalized.dropna(subset=['lastOnlineTs'], inplace=True)
db_merged_normalized['lastOnlineTs'] = pandas.to_numeric(db_merged_normalized['lastOnlineTs'])

db_merged_normalized['lastOnlineTime'].replace('', np.nan, inplace=True)
db_merged_normalized.dropna(subset=['lastOnlineTime'], inplace=True)
db_merged_normalized['lastOnlineTime'] = pandas.to_numeric(db_merged_normalized['lastOnlineTime'])

db_merged_normalized['distance_y'].replace('', np.nan, inplace=True)
db_merged_normalized.dropna(subset=['distance_y'], inplace=True)

for column in integer_columns: 
    print(column)
    db_merged_normalized[column] = (db_merged_normalized[column] - db_merged_normalized[column].min()) / (db_merged_normalized[column].max() - db_merged_normalized[column].min())

db_merged_normalized.to_excel('db_merged_normalized.xlsx')


