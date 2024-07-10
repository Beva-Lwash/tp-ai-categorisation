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
df_merged = pandas.merge(object_df_table, api_df_table, on="userId", how='inner', suffixes=('', '_y')).filter(regex='^(?!.*_y)')






date_columns = ['lastOnline', 'lastOnlineDate']
number_columns = ['age', 'counts_pictures', 'counts_profileVisits', 'counts_kisses', 'lastOnlineTs', 'lang_count', 'countDetails', 'distance', 'counts_details', 'counts_fans', 'counts_g', 'lastOnlineTime']
binary_columns = ['isFlirtstar', 'isHighlighted', 'isInfluencer', 'isMobile', 'isNew', 'isOnline', 'isVip', 'verified', 'shareProfileEnabled', 'birthd']
boolean_columns = ['flirtInterests_chat', 'flirtInterests_friends', 'flirtInterests_date', 'connectedToFacebook', 'isVIP', 'isVerified', 'lang_fr', 'lang_en', 'lang_de', 'lang_it', 'lang_es', 'lang_pt', 'crypt', 'flirtstar', 'freshman', 'hasBirthday', 'highlighted', 'locked', 'mobile', 'online', 'isSystemProfile']
string_columns = ['name', 'whazzup', 'freetext']
enum_columns = ['city', 'locationCity', 'genderLooking', 'location', 'pictureId']
empty_columns = ['locationCitySub', 'userInfo_visitDate']
id_columns = ['userId']

df_merged_dropped = df_merged.drop(columns = date_columns + string_columns + empty_columns + id_columns)

for column in binary_columns:
    df_merged_dropped[column] = df_merged_dropped[column].astype(bool)


df_merged_dropped.to_excel('df_merged_dropped.xlsx')

"""
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

"""