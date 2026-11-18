"""
IFT 799 - Science des données
TP1 - Préparation des données

Étape 2 :
- supprimer les variables constantes;
- calculer la variance des variables restantes;
- identifier les variables les plus variables.
"""

from pathlib import Path

import pandas as pd



# 1. Chemins


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "TCGA-PANCAN-HiSeq-801x20531"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"

DATA_FILE = DATA_DIR / "data.csv"
LABELS_FILE = DATA_DIR / "labels.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# 2. Chargement


print("=" * 70)
print("IFT799 - TP1 : PRÉPARATION DES DONNÉES")
print("=" * 70)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])



# 3. Suppression des variables constantes


variances = X.var(axis=0)

constant_features = variances[variances == 0].index.tolist()

X_filtered = X.drop(columns=constant_features)

print("\n1. SUPPRESSION DES VARIABLES CONSTANTES")
print("-" * 70)

print(f"Variables initiales : {X.shape[1]}")
print(f"Variables constantes supprimées : {len(constant_features)}")
print(f"Variables restantes : {X_filtered.shape[1]}")



# 4. Classement selon la variance


filtered_variances = X_filtered.var(axis=0)

variance_ranking = (
    filtered_variances
    .sort_values(ascending=False)
    .rename("variance")
    .reset_index()
    .rename(columns={"index": "gene"})
)

print("\n2. VARIABLES AVEC LA PLUS GRANDE VARIANCE")
print("-" * 70)

print(variance_ranking.head(30).to_string(index=False))



# 5. Export du classement


ranking_file = OUTPUT_DIR / "gene_variance_ranking.csv"

variance_ranking.to_csv(
    ranking_file,
    index=False
)

print("\n3. FICHIER CRÉÉ")
print("-" * 70)

print(ranking_file)



# 6. Résumé


print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)

print(f"Patients : {X_filtered.shape[0]}")
print(f"Variables retenues après filtrage : {X_filtered.shape[1]}")
print(f"Variables constantes supprimées : {len(constant_features)}")

print("\nPréparation terminée.")