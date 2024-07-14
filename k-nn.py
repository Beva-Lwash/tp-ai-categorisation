import csv
import pandas as pandas
import sqlite3
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

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

for column in binary_columns + boolean_columns: #ici on change les 0 et 1 en booleans et les true false en booleans
    df[column] = df[column].astype(bool)

df = df.T.drop_duplicates().T  #ici on elimine les columns duplicate dans leurs valeurs

df['lastOnlineTs'].replace('', np.nan, inplace=True) #ici on elimine les rangees vides et on change les timestamp notation scientifique en integers
df.dropna(subset=['lastOnlineTs'], inplace=True)
df['lastOnlineTs'] = pandas.to_numeric(df['lastOnlineTs'])

for column in number_columns: # ici on normalise les columns qui ont un number
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())


    X = df.drop("est_populaire", axis=1)
    Y = df['est_populaire'].astype(bool)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

"""
Fonction qui créer un classificateur k-nn
"""
def knnClass(k,X_train,Y_train):
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(X_train, np.ravel(Y_train))
  return clf

"""
# Fonction qui génére et affiche une matrice de confusion à partir des données 
# de validations
"""
def confusionMatrixknn(knnClass,X_train,Y_train,X_test,Y_test):
  metrics.plot_confusion_matrix(knnClass, X_test, Y_test)
  print("Train accuracy: ", knnClass.score(X_train, Y_train))
  print("Test Accuracy: ", knnClass.score(X_test, Y_test))

"""
le k actuel que l'on a besoin appellé i 
"""

temp=math.sqrt(len(Y_train.axes[0])/2) #ici j'ai le  k actuel fais la formule du court k = racine (n/c)

def plus_proche_nombre_impair(n):
  entier_plus_proche =round(n)

  if entier_plus_proche % 2 !=0:
    return entier_plus_proche
  else:
    if n > entier_plus_proche:
      return entier_plus_proche+1
    else:
      return entier_plus_proche-1

i=plus_proche_nombre_impair(temp)
glassknn = knnClass(i,X_train, Y_train)
confusionMatrixknn(glassknn,X_train, Y_train, X_test, Y_test)