import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Charger les données
api_data = pd.read_csv('lovoo_v3_users_api-results.csv')
instance_data = pd.read_csv('lovoo_v3_users_instances.csv')

# Inclure les colonnes supplémentaires counts_pictures et counts_fans
columns_to_keep_api = ['userId', 'name', 'age', 'gender', 'city', 'counts_profileVisits', 'counts_kisses', 'counts_fans']
columns_to_keep_instance = ['userId', 'name', 'age', 'gender', 'city', 'counts_profileVisits', 'counts_kisses', 'flirtInterests_chat', 'flirtInterests_friends', 'flirtInterests_date', 'locked', 'counts_pictures']

# Filtrer les colonnes pertinentes
api_data_filtered = api_data[columns_to_keep_api]
instance_data_filtered = instance_data[columns_to_keep_instance]

# Gérer les identifiants utilisateur en double
merged_data = pd.merge(api_data_filtered, instance_data_filtered, on='userId', suffixes=('_api', '_instance'), how='outer')

# Choisir les valeurs de api_data pour les colonnes en chevauchement
for col in columns_to_keep_api + ['counts_pictures']:
    if col != 'userId' and f'{col}_api' in merged_data.columns and f'{col}_instance' in merged_data.columns:
        merged_data[col] = merged_data[f'{col}_api'].combine_first(merged_data[f'{col}_instance'])
    elif f'{col}_api' in merged_data.columns:
        merged_data[col] = merged_data[f'{col}_api']
    elif f'{col}_instance' in merged_data.columns:
        merged_data[col] = merged_data[f'{col}_instance']

# Filtrer les comptes bloqués
if 'locked' in merged_data.columns:
    merged_data = merged_data[merged_data['locked'] != True]

# Supprimer les colonnes intermédiaires
merged_data = merged_data[['userId', 'name', 'age', 'gender', 'city', 'counts_profileVisits', 'counts_kisses', 'counts_fans', 'counts_pictures', 'flirtInterests_chat', 'flirtInterests_friends', 'flirtInterests_date']]

# Sauvegarder le fichier de données fusionnées
merged_data.to_csv('merged_data.csv', index=False)

# Préparation des données
# Transformer les colonnes catégorielles en variables numériques
merged_data['gender'] = merged_data['gender'].map({'M': 0, 'F': 1})
merged_data = pd.get_dummies(merged_data, columns=['flirtInterests_chat', 'flirtInterests_friends', 'flirtInterests_date'], drop_first=True)

# Gérer les valeurs manquantes en remplissant avec des valeurs par défaut ou en supprimant les lignes avec des valeurs manquantes
merged_data = merged_data.dropna()

# Définir les caractéristiques (features) et la cible (target)
X = merged_data.drop(['userId', 'name', 'city'], axis=1)
y = (merged_data['counts_profileVisits'] > 10) & (merged_data['counts_kisses'] / merged_data['counts_profileVisits'] > 0.05)
y = y.astype(int)  # Convertir en entiers (0 ou 1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = clf.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# Définir les hyperparamètres à optimiser
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Initialiser le modèle Random Forest
rf = RandomForestClassifier(random_state=42)

# Initialiser GridSearchCV avec validation croisée à 5 plis
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Exécuter GridSearchCV
grid_search.fit(X_train, y_train)

# Meilleurs hyperparamètres trouvés
best_params = grid_search.best_params_
print("Meilleurs hyperparamètres : ", best_params)

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle final
best_rf = grid_search.best_estimator_

# Prédire sur l'ensemble de test
y_pred_best = best_rf.predict(X_test)

# Évaluation du modèle optimisé
accuracy_best = accuracy_score(y_test, y_pred_best)
classification_rep_best = classification_report(y_test, y_pred_best)

print(f"Accuracy après optimisation: {accuracy_best}")
print("Classification Report après optimisation:")
print(classification_rep_best)

# Afficher les résultats
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)



# Importance des caractéristiques
importances = best_rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Tracer l'importance des caractéristiques
plt.figure(figsize=(12, 6))
plt.title("Importance des caractéristiques")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()



# Prédire sur l'ensemble de test avec le meilleur modèle
y_pred_best = best_rf.predict(X_test)

# Créer la matrice de confusion
cm = confusion_matrix(y_test, y_pred_best)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.show()
