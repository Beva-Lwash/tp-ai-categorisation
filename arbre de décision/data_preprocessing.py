import csv
import pandas as pd
import sqlite3
import numpy as np

# Charger les données CSV
def loadData(file):
    return pd.read_csv(file, header=0)

api_df_csv = loadData('lovoo_v3_users_api-results.csv')
object_df_csv = loadData('lovoo_v3_users_instances.csv')
csv_merged = pd.merge(api_df_csv, object_df_csv, on="userId")
csv_merged.sort_index(inplace=True)

# Connexion à la base de données
dbfile = 'users_lovoo_v3.db'
con = sqlite3.connect(dbfile)

# Charger les données de la base de données
object_df_table = pd.read_sql_query("SELECT * FROM objects", con)
api_df_table = pd.read_sql_query("SELECT * FROM api", con)

# Fusionner les données
df = pd.merge(object_df_table, api_df_table, on="userId", how='inner', suffixes=('', '_y')).filter(regex='^(?!.*_y)')

# Définir la popularité de l'utilisateur
def pop_score(row):
    return (row['counts_profileVisits'] > 10) and ((row['counts_kisses'] / row['counts_profileVisits']) > 0.05)
df['est_populaire'] = df.apply(pop_score, axis=1)  # Créer une colonne indiquant si l'utilisateur est populaire

# Nettoyage des données, suppression des colonnes non nécessaires
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

df = df.drop(columns = date_columns + string_columns + enum_columns + empty_columns + id_columns + single_value_columns + outdated_columns)

# Conversion des types de données
for column in binary_columns + boolean_columns:
    df[column] = df[column].astype(bool)

# Suppression des colonnes dupliquées
df = df.T.drop_duplicates().T

# Remplacer les valeurs vides et convertir les types de données
df['lastOnlineTs'].replace('', np.nan, inplace=True)
df.dropna(subset=['lastOnlineTs'], inplace=True)
df['lastOnlineTs'] = pd.to_numeric(df['lastOnlineTs'])

# Normalisation des colonnes numériques
for column in number_columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Sauvegarder les données traitées dans une nouvelle table de la base de données
df.to_sql('processed_data', con, if_exists='replace', index=False)

# Fermer la connexion à la base de données
con.close()
