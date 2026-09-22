"""
IFT 799 - Science des données
TP1 - Comparaison de sous-ensembles de variables

Objectif :
Comparer plusieurs tailles de sous-ensembles de gènes à forte variance
en utilisant directement le critère de séparation défini dans le TP :

- distance intra-classe
- distance inter-classe
- Overlap

Pour cette étape exploratoire, on utilise la distance euclidienne.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
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



# 2. Chargement des données


print("=" * 75)
print("IFT799 - TP1 : COMPARAISON DES SOUS-ENSEMBLES")
print("=" * 75)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])

y = labels["Class"]



# 3. Retirer les variables constantes et les classer
#    selon leur variance


variances = X.var(axis=0)

non_constant_variances = variances[variances > 0]

ranked_genes = (
    non_constant_variances
    .sort_values(ascending=False)
    .index
    .tolist()
)

print("\nVariables non constantes :", len(ranked_genes))



# 4. Fonctions de distance


def euclidean_distance(a, b):
    """
    Distance euclidienne entre deux vecteurs.
    """
    return np.linalg.norm(a - b)


def class_center(class_data):
    """
    Centre d'une classe = moyenne des observations.
    """
    return class_data.mean(axis=0)


def intra_class_distance(class_data, center):
    """
    Distance intra-classe selon la définition du TP :
    distance maximale entre un objet de la classe
    et le centre de cette classe.
    """

    distances = np.linalg.norm(
        class_data - center,
        axis=1
    )

    return distances.max()


def directed_inter_distance(class_data, other_center):
    """
    Distance minimale entre un objet d'une classe
    et le centre de l'autre classe.
    """

    distances = np.linalg.norm(
        class_data - other_center,
        axis=1
    )

    return distances.min()


def inter_class_distance(
    class1_data,
    class2_data,
    center1,
    center2
):
    """
    Distance inter-classe selon la définition du TP :

    min(
        distance minimale C1 -> centre C2,
        distance minimale C2 -> centre C1
    )
    """

    d12 = directed_inter_distance(
        class1_data,
        center2
    )

    d21 = directed_inter_distance(
        class2_data,
        center1
    )

    return min(d12, d21)



# 5. Comparaison des tailles


subset_sizes = [50, 100, 200, 500, 1000]

classes = sorted(y.unique())

summary_results = []
detail_results = []


for size in subset_sizes:

    print("\n" + "=" * 75)
    print(f"SOUS-ENSEMBLE : TOP {size} GÈNES")
    print("=" * 75)

    selected_genes = ranked_genes[:size]

    X_subset = X[selected_genes].to_numpy()


    # -----------------------------------------------------
    # Séparer les données par classe
    # -----------------------------------------------------

    class_data = {}

    centers = {}

    intra_distances = {}

    for cancer_class in classes:

        mask = (y == cancer_class).to_numpy()

        current_data = X_subset[mask]

        center = class_center(current_data)

        intra = intra_class_distance(
            current_data,
            center
        )

        class_data[cancer_class] = current_data
        centers[cancer_class] = center
        intra_distances[cancer_class] = intra


    # -----------------------------------------------------
    # Toutes les paires de classes
    # -----------------------------------------------------

    overlaps = []

    for class1, class2 in combinations(classes, 2):

        inter = inter_class_distance(
            class_data[class1],
            class_data[class2],
            centers[class1],
            centers[class2]
        )

        overlap = (
            intra_distances[class1]
            + intra_distances[class2]
        ) / (2 * inter)

        overlaps.append(overlap)

        detail_results.append(
            {
                "nombre_variables": size,
                "classe_1": class1,
                "classe_2": class2,
                "dist_intra_1": intra_distances[class1],
                "dist_intra_2": intra_distances[class2],
                "dist_inter": inter,
                "overlap": overlap,
                "bien_separees": overlap < 1
            }
        )


    # -----------------------------------------------------
    # Résumé du sous-ensemble
    # -----------------------------------------------------

    overlaps = np.array(overlaps)

    well_separated = np.sum(overlaps < 1)

    summary_results.append(
        {
            "nombre_variables": size,
            "overlap_moyen": overlaps.mean(),
            "overlap_median": np.median(overlaps),
            "overlap_min": overlaps.min(),
            "overlap_max": overlaps.max(),
            "paires_bien_separees": well_separated,
            "total_paires": len(overlaps)
        }
    )

    print(f"Overlap moyen      : {overlaps.mean():.4f}")
    print(f"Overlap médian     : {np.median(overlaps):.4f}")
    print(f"Overlap minimum    : {overlaps.min():.4f}")
    print(f"Overlap maximum    : {overlaps.max():.4f}")

    print(
        f"Paires avec Overlap < 1 : "
        f"{well_separated}/{len(overlaps)}"
    )



# 6. Tableau résumé


summary_df = pd.DataFrame(summary_results)

detail_df = pd.DataFrame(detail_results)


print("\n" + "=" * 75)
print("COMPARAISON FINALE")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "overlap_moyen": "{:.4f}".format,
            "overlap_median": "{:.4f}".format,
            "overlap_min": "{:.4f}".format,
            "overlap_max": "{:.4f}".format
        }
    )
)



# 7. Export


summary_file = (
    OUTPUT_DIR
    / "feature_subset_comparison.csv"
)

detail_file = (
    OUTPUT_DIR
    / "feature_subset_pairwise_details.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

detail_df.to_csv(
    detail_file,
    index=False
)


print("\nFichiers créés :")
print(summary_file)
print(detail_file)

print("\nAnalyse terminée.")