import pandas as pd
import numpy as np
import sqlite3
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données traitées à partir de la base de données
dbfile = 'users_lovoo_v3.db'
con = sqlite3.connect(dbfile)
df = pd.read_sql_query("SELECT * FROM processed_data", con)
con.close()

# Diviser le jeu de données
X = df.drop('est_populaire', axis=1)
y = df['est_populaire'].astype(int)  # S'assurer que la variable cible est de type entier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle
tree_model = DecisionTreeClassifier(random_state=42, criterion='gini')
tree_model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Afficher les résultats
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Visualisation de la matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=tree_model.classes_)
disp = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Popular', 'Popular'], yticklabels=['Not Popular', 'Popular'])
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de Confusion')
plt.savefig('confusion_matrix.png')
plt.show()

# Visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, class_names=['Not Popular', 'Popular'], filled=True)
plt.title('Arbre de Décision')
plt.savefig('decision_tree.png')
plt.show()

# Calcul de l'indice de Gini
n_nodes = tree_model.tree_.node_count
children_left = tree_model.tree_.children_left
children_right = tree_model.tree_.children_right
feature = tree_model.tree_.feature
threshold = tree_model.tree_.threshold
impurity = tree_model.tree_.impurity
n_node_samples = tree_model.tree_.n_node_samples

print(f"{'Node':>10} {'Feature':>15} {'Threshold':>10} {'Impurity':>10} {'Samples':>10}")
for i in range(n_nodes):
    if children_left[i] == children_right[i]:  # feuille
        print(f"{i:>10} {'feuille':>15} {'':>10} {impurity[i]:>10.2f} {n_node_samples[i]:>10}")
    else:
        print(f"{i:>10} {X.columns[feature[i]]:>15} {threshold[i]:>10.2f} {impurity[i]:>10.2f} {n_node_samples[i]:>10}")
