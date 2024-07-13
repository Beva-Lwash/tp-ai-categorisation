import csv
import pandas as pandas
import sqlite3
import numpy as np
from sklearn.naive_bayes import GaussianNB

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
df = pandas.merge(object_df_table, api_df_table, on="userId", how='inner', suffixes=('', '_y')).filter(regex='^(?!.*_y)')

#df['diff'] = df['lastOnlineTime'] - df['lastOnlineTs']  #ici on verifie entre lastOnlineTs et lastOnlineTime quel est le plus recent (cest lastOnlineTs alors on decide de la garder)

date_columns = ['lastOnline', 'lastOnlineDate']
number_columns = ['age', 'counts_pictures', 'counts_profileVisits', 'counts_kisses', 'lastOnlineTs', 'lang_count', 'countDetails', 'distance', 'counts_details', 'counts_fans', 'counts_g']
binary_columns = ['isFlirtstar', 'isHighlighted', 'isInfluencer', 'isMobile', 'isNew', 'isOnline', 'isVip', 'verified', 'shareProfileEnabled', 'birthd']
boolean_columns = ['flirtInterests_chat', 'flirtInterests_friends', 'flirtInterests_date', 'connectedToFacebook', 'isVIP', 'isVerified', 'lang_fr', 'lang_en', 'lang_de', 'lang_it', 'lang_es', 'lang_pt', 'crypt', 'flirtstar', 'freshman', 'hasBirthday', 'highlighted', 'locked', 'mobile', 'online', 'isSystemProfile']
string_columns = ['name', 'whazzup', 'freetext']
enum_columns = ['city', 'locationCity', 'genderLooking', 'location', 'pictureId']
empty_columns = ['locationCitySub', 'userInfo_visitDate']
id_columns = ['userId']
single_value_columns = ['gender']
outdated_columns = ['lastOnlineTime']

df = df.drop(columns = date_columns + string_columns + empty_columns + id_columns + single_value_columns + outdated_columns) #ici on enleve les columns pas interessantes

for column in binary_columns: #ici on change les 0 et 1 en true et false
    df[column] = df[column].astype(bool)

df['lastOnlineTs'].replace('', np.nan, inplace=True) #ici on elimine les rangees vides et on change les timestamp notation scientifique en integers
df.dropna(subset=['lastOnlineTs'], inplace=True)
df['lastOnlineTs'] = pandas.to_numeric(df['lastOnlineTs'])

df = df.loc[:,~df.apply(lambda x: x.duplicated(),axis=1).all()].copy() #ici on elimine les columns duplicate dans leurs valeurs

for column in number_columns: # ici on normalise les columns qui ont un number
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

df.to_excel('df.xlsx')


