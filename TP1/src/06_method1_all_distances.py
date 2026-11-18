"""
IFT 799 - Science des données
TP1 - Méthode 1

Calcul de la séparation entre toutes les paires de classes
avec les trois métriques demandées :

1. Distance euclidienne
2. Distance de Mahalanobis
3. Distance cosinus

Sous-ensemble retenu :
100 gènes non constants ayant les plus grandes variances.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine



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


print("=" * 78)
print("IFT799 - TP1 : MÉTHODE 1")
print("=" * 78)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])
y = labels["Class"]



# 3. Sélection des 100 gènes


variances = X.var(axis=0)

non_constant_variances = variances[variances > 0]

selected_genes = (
    non_constant_variances
    .sort_values(ascending=False)
    .head(100)
    .index
    .tolist()
)

X_selected = X[selected_genes].to_numpy(dtype=float)

classes = sorted(y.unique())


print("\nSous-ensemble utilisé : 100 gènes")
print("Nombre de patients :", X_selected.shape[0])
print("Nombre de variables :", X_selected.shape[1])



# 4. Organisation par classe


class_data = {}
class_centers = {}

for cancer_class in classes:

    mask = (y == cancer_class).to_numpy()

    current_data = X_selected[mask]

    class_data[cancer_class] = current_data

    class_centers[cancer_class] = current_data.mean(axis=0)



# 5. Matrice de covariance pour Mahalanobis


covariance_matrix = np.cov(
    X_selected,
    rowvar=False
)

condition_number = np.linalg.cond(covariance_matrix)

print("\nCondition de la matrice de covariance :")
print(f"{condition_number:.4e}")


# Utilisation de l'inverse lorsque possible.
# La pseudo-inverse est utilisée comme solution robuste
# si la matrice est numériquement mal conditionnée.

if np.isfinite(condition_number) and condition_number < 1e12:

    inverse_covariance = np.linalg.inv(covariance_matrix)

    covariance_method = "inverse"

else:

    inverse_covariance = np.linalg.pinv(covariance_matrix)

    covariance_method = "pseudo-inverse"


print(
    "Méthode utilisée pour la covariance :",
    covariance_method
)



# 6. Fonctions de distance


def euclidean_distance(a, b):

    return np.linalg.norm(a - b)


def mahalanobis_distance(a, b):

    difference = a - b

    squared_distance = (
        difference.T
        @ inverse_covariance
        @ difference
    )

    return float(np.sqrt(max(squared_distance, 0)))


def cosine_distance(a, b):

    # scipy.spatial.distance.cosine retourne :
    # 1 - similarité cosinus

    return float(cosine(a, b))



# 7. Calcul d'une métrique complète


def calculate_metric(metric_name, distance_function):

    intra_distances = {}

    # Distance intra-classe
    for cancer_class in classes:

        current_data = class_data[cancer_class]

        center = class_centers[cancer_class]

        distances = [
            distance_function(patient, center)
            for patient in current_data
        ]

        intra_distances[cancer_class] = max(distances)


    results = []

    # Distance inter-classe + Overlap
    for class1, class2 in combinations(classes, 2):

        data1 = class_data[class1]
        data2 = class_data[class2]

        center1 = class_centers[class1]
        center2 = class_centers[class2]


        distances_1_to_2 = [
            distance_function(patient, center2)
            for patient in data1
        ]

        distances_2_to_1 = [
            distance_function(patient, center1)
            for patient in data2
        ]


        inter_distance = min(
            min(distances_1_to_2),
            min(distances_2_to_1)
        )


        overlap = (
            intra_distances[class1]
            + intra_distances[class2]
        ) / (
            2 * inter_distance
        )


        results.append(
            {
                "metrique": metric_name,
                "classe_1": class1,
                "classe_2": class2,
                "dist_intra_1": intra_distances[class1],
                "dist_intra_2": intra_distances[class2],
                "dist_inter": inter_distance,
                "overlap": overlap,
                "bien_separees": overlap < 1
            }
        )

    return results



# 8. Exécution des trois métriques


all_results = []


print("\nCalcul distance euclidienne...")

all_results.extend(
    calculate_metric(
        "Euclidienne",
        euclidean_distance
    )
)


print("Calcul distance de Mahalanobis...")

all_results.extend(
    calculate_metric(
        "Mahalanobis",
        mahalanobis_distance
    )
)


print("Calcul distance cosinus...")

all_results.extend(
    calculate_metric(
        "Cosinus",
        cosine_distance
    )
)



# 9. Résultats détaillés


results_df = pd.DataFrame(all_results)


for metric in [
    "Euclidienne",
    "Mahalanobis",
    "Cosinus"
]:

    print("\n" + "=" * 78)
    print(metric.upper())
    print("=" * 78)

    metric_df = (
        results_df[
            results_df["metrique"] == metric
        ]
        .sort_values("overlap")
    )

    print(
        metric_df[
            [
                "classe_1",
                "classe_2",
                "dist_intra_1",
                "dist_intra_2",
                "dist_inter",
                "overlap",
                "bien_separees"
            ]
        ].to_string(
            index=False,
            formatters={
                "dist_intra_1": "{:.6f}".format,
                "dist_intra_2": "{:.6f}".format,
                "dist_inter": "{:.6f}".format,
                "overlap": "{:.6f}".format
            }
        )
    )



# 10. Tableau comparatif des Overlap


comparison = results_df.pivot(
    index=["classe_1", "classe_2"],
    columns="metrique",
    values="overlap"
).reset_index()


print("\n" + "=" * 78)
print("COMPARAISON DES OVERLAP")
print("=" * 78)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Euclidienne": "{:.6f}".format,
            "Mahalanobis": "{:.6f}".format,
            "Cosinus": "{:.6f}".format
        }
    )
)



# 11. Résumé


summary = (
    results_df
    .groupby("metrique")
    .agg(
        overlap_moyen=("overlap", "mean"),
        overlap_median=("overlap", "median"),
        overlap_min=("overlap", "min"),
        overlap_max=("overlap", "max"),
        paires_bien_separees=("bien_separees", "sum")
    )
    .reset_index()
)


print("\n" + "=" * 78)
print("RÉSUMÉ PAR MÉTRIQUE")
print("=" * 78)

print(
    summary.to_string(
        index=False,
        formatters={
            "overlap_moyen": "{:.6f}".format,
            "overlap_median": "{:.6f}".format,
            "overlap_min": "{:.6f}".format,
            "overlap_max": "{:.6f}".format
        }
    )
)



# 12. Export


details_file = (
    OUTPUT_DIR
    / "method1_all_distances.csv"
)

comparison_file = (
    OUTPUT_DIR
    / "method1_overlap_comparison.csv"
)

summary_file = (
    OUTPUT_DIR
    / "method1_summary.csv"
)


results_df.to_csv(
    details_file,
    index=False
)

comparison.to_csv(
    comparison_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)


print("\nFichiers créés :")

print(details_file)
print(comparison_file)
print(summary_file)

print("\nMéthode 1 terminée.")