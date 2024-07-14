import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Database connection and data fetching
dbfile = 'users_lovoo_v3.db'
con = sqlite3.connect(dbfile)
cur = con.cursor()
object_df_table = pd.read_sql_query("SELECT * FROM objects", con)
api_df_table = pd.read_sql_query("SELECT * FROM api", con)
con.close()
df = pd.merge(object_df_table, api_df_table, on="userId", how='inner', suffixes=('', '_y')).filter(regex='^(?!.*_y)')

# Define popularity score
def pop_score(row):
    return (row['counts_profileVisits'] > 10) and ((row['counts_kisses'] / row['counts_profileVisits']) > 0.05)
df['est_populaire'] = df.apply(pop_score, axis=1)

# Remove unnecessary columns
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

df = df.drop(columns=date_columns + string_columns + enum_columns + empty_columns + id_columns + single_value_columns + outdated_columns)

# Convert binary and boolean columns to boolean type
for column in binary_columns + boolean_columns:
    df[column] = df[column].astype(bool)

# Remove duplicate columns
df = df.T.drop_duplicates().T

# Handle missing values and convert to numeric
df['lastOnlineTs'].replace('', np.nan, inplace=True)
df.dropna(subset=['lastOnlineTs'], inplace=True)
df['lastOnlineTs'] = pd.to_numeric(df['lastOnlineTs'])

# Normalize number columns
for column in number_columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())


# Create KNN classifier
def knnClass(k, X_train, Y_train):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, np.ravel(Y_train))
    return clf

# Generate and return confusion matrix
def confusionMatrixknn(knnClass, X_test, Y_test):
    y_pred = knnClass.predict(X_test)
    return confusion_matrix(Y_test, y_pred)

# Find nearest odd number
def plus_proche_nombre_impair(n):
    entier_plus_proche = round(n)
    if entier_plus_proche % 2 != 0:
        return entier_plus_proche
    else:
        return entier_plus_proche + 1 if n > entier_plus_proche else entier_plus_proche - 1

# Generate confusion matrix
def knn_confusion(df, seed):
    X = df.drop("est_populaire", axis=1)
    y = df['est_populaire'].astype(bool)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    temp = math.sqrt(len(y_train) / 2)
    j = plus_proche_nombre_impair(temp)
    knn = knnClass(j, X_train, y_train)
    return confusionMatrixknn(knn, X_test, y_test)

# Get average confusion matrix
def get_average_confusion_matrix(df):
    iterations = 1000
    cm_list = []
    for i in range(iterations):
        cm_list.append(knn_confusion(df, i))
    return np.mean(cm_list, axis=0)

# Plot average confusion matrix
cm = get_average_confusion_matrix(df)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non", "Oui"])
disp.plot(values_format='.0f')
plt.title('Matrice de confusion')
plt.xlabel('Valeur prédite')
plt.ylabel('Valeur test')
plt.show()