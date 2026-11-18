"""
IFT 799 - Science des données
TP1 - Compréhension et visualisation des données

Étape 1 : Exploration initiale du jeu de données
"""

from pathlib import Path

import numpy as np
import pandas as pd



# 1. Chemins


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "TCGA-PANCAN-HiSeq-801x20531"
)

DATA_FILE = DATA_DIR / "data.csv"
LABELS_FILE = DATA_DIR / "labels.csv"



# 2. Chargement des données


print("=" * 70)
print("IFT799 - TP1 : EXPLORATION INITIALE")
print("=" * 70)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

print("\n1. DIMENSIONS DES FICHIERS")
print("-" * 70)

print(
    f"data.csv   : {data.shape[0]} lignes x "
    f"{data.shape[1]} colonnes"
)

print(
    f"labels.csv : {labels.shape[0]} lignes x "
    f"{labels.shape[1]} colonnes"
)



# 3. Aperçu


print("\n2. PREMIÈRES LIGNES DE data.csv")
print("-" * 70)
print(data.iloc[:5, :8])

print("\n3. PREMIÈRES LIGNES DE labels.csv")
print("-" * 70)
print(labels.head())



# 4. Identifier les colonnes d'identification


id_column_data = data.columns[0]
id_column_labels = labels.columns[0]

print("\n4. IDENTIFIANTS")
print("-" * 70)

print(f"Colonne ID dans data.csv   : {id_column_data}")
print(f"Colonne ID dans labels.csv : {id_column_labels}")



# 5. Variables génétiques


X = data.drop(columns=[id_column_data])

print("\n5. VARIABLES GÉNÉTIQUES")
print("-" * 70)

print(f"Nombre de patients : {X.shape[0]}")
print(f"Nombre de variables génétiques : {X.shape[1]}")



# 6. Distribution des classes


class_column = "Class"

print("\n6. DISTRIBUTION DES CLASSES")
print("-" * 70)

class_counts = labels[class_column].value_counts()

print(class_counts.sort_index())

print(
    f"\nNombre de classes : "
    f"{labels[class_column].nunique()}"
)

print(
    "Classes :",
    sorted(labels[class_column].unique().tolist())
)



# 7. Valeurs manquantes


print("\n7. VALEURS MANQUANTES")
print("-" * 70)

missing_data = X.isna().sum().sum()
missing_labels = labels.isna().sum().sum()

print(
    f"Valeurs manquantes dans les variables génétiques : "
    f"{missing_data}"
)

print(
    f"Valeurs manquantes dans labels.csv : "
    f"{missing_labels}"
)



# 8. Valeurs infinies


print("\n8. VALEURS INFINIES")
print("-" * 70)

numeric_array = X.to_numpy()

number_inf = np.isinf(numeric_array).sum()

print(f"Nombre de valeurs infinies : {number_inf}")



# 9. Variables constantes


print("\n9. VARIABLES CONSTANTES")
print("-" * 70)

variances = X.var(axis=0)

constant_features = variances[variances == 0]

print(
    f"Nombre de variables constantes : "
    f"{len(constant_features)}"
)

if len(constant_features) > 0:
    print("Premières variables constantes :")
    print(constant_features.index[:20].tolist())



# 10. Variance


print("\n10. STATISTIQUES DE VARIANCE")
print("-" * 70)

print(variances.describe())

print("\n10 variables avec la plus grande variance :")
print(
    variances
    .sort_values(ascending=False)
    .head(10)
)



# 11. Alignement des identifiants


print("\n11. CORRESPONDANCE DES PATIENTS")
print("-" * 70)

same_ids = (
    data[id_column_data]
    .astype(str)
    .reset_index(drop=True)
    ==
    labels[id_column_labels]
    .astype(str)
    .reset_index(drop=True)
).all()

print(
    f"Les identifiants correspondent ligne par ligne : "
    f"{same_ids}"
)



# 12. Résumé


print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)

print(f"Patients                 : {X.shape[0]}")
print(f"Variables génétiques     : {X.shape[1]}")
print(
    f"Classes                  : "
    f"{labels[class_column].nunique()}"
)
print(f"Valeurs manquantes       : {missing_data}")
print(f"Valeurs infinies         : {number_inf}")
print(
    f"Variables constantes     : "
    f"{len(constant_features)}"
)
print(
    f"IDs correctement alignés : "
    f"{same_ids}"
)

print("\nExploration initiale terminée.")