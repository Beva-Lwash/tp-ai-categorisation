import csv
import pandas as pandas
import sqlite3
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

def pop_score(row):
    return (row['counts_profileVisits'] > 10) and ((row['counts_kisses'] / row['counts_profileVisits']) > 0.05)
df['est_populaire'] = df.apply(pop_score, axis=1) #ici on cree une colonne qui indique la popularite du profil

#df['diff'] = df['lastOnlineTime'] - df['lastOnlineTs']  #ici on verifie entre lastOnlineTs et lastOnlineTime quel est le plus recent (cest lastOnlineTs alors on decide de la garder)

date_columns = ['lastOnline', 'lastOnlineDate']
number_columns = ['age', 'counts_pictures', 'counts_profileVisits', 'counts_kisses', 'lastOnlineTs', 'lang_count', 'countDetails', 'distance', 'counts_details', 'counts_fans', 'counts_g']
binary_columns = ['isFlirtstar', 'isHighlighted', 'isInfluencer', 'isMobile', 'isNew', 'isOnline', 'isVip', 'verified', 'shareProfileEnabled', 'birthd']
boolean_columns = ['flirtInterests_chat', 'flirtInterests_friends', 'flirtInterests_date', 'connectedToFacebook', 'isVIP', 'isVerified', 'lang_fr', 'lang_en', 'lang_de', 'lang_it', 'lang_es', 'lang_pt', 'crypt', 'flirtstar', 'freshman', 'hasBirthday', 'highlighted', 'locked', 'mobile', 'online', 'isSystemProfile']
string_columns = ['name', 'whazzup', 'freetext']
enum_columns = ['city', 'locationCity', 'country', 'genderLooking', 'location', 'pictureId']
empty_columns = ['locationCitySub', 'userInfo_visitDate']
id_columns = ['userId']
single_value_columns = ['gender']
outdated_columns = ['lastOnlineTime']

df = df.drop(columns = date_columns + string_columns + enum_columns + empty_columns + id_columns + single_value_columns + outdated_columns) #ici on enleve les columns pas interessantes

for column in binary_columns + boolean_columns: #ici on change les 0 et 1 en booleans
    df[column] = df[column].astype(bool)

df = df.T.drop_duplicates().T  #ici on elimine les columns duplicate dans leurs valeurs

df['lastOnlineTs'].replace('', np.nan, inplace=True) #ici on elimine les rangees vides et on change les timestamp notation scientifique en integers
df.dropna(subset=['lastOnlineTs'], inplace=True)
df['lastOnlineTs'] = pandas.to_numeric(df['lastOnlineTs'])

for column in number_columns: # ici on normalise les columns qui ont un number
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

def gaussian_bayesian(df):
    X = df.drop("est_populaire", axis=1)
    y = df['est_populaire'].astype(bool)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of well labeled points out of a total %d points : %d" % (X_test.shape[0], (y_test == y_pred).sum()))
gaussian_bayesian(df)