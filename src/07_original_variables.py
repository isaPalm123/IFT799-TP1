"""
IFT 799 - Science des données
TP1 - Méthode 2 : variables originales

Objectif :
Pour chaque paire de classes :
1. choisir un premier gène minimisant l'Overlap en 1D;
2. choisir un deuxième gène, compte tenu du premier,
   minimisant l'Overlap en 2D;
3. produire un nuage de points.

La recherche est gloutonne (greedy) et non exhaustive,
ce qui respecte l'énoncé du TP.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 1. Chemins


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "TCGA-PANCAN-HiSeq-801x20531"
)

FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"

DATA_FILE = DATA_DIR / "data.csv"
LABELS_FILE = DATA_DIR / "labels.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)



# 2. Chargement


print("=" * 78)
print("IFT799 - TP1 : MÉTHODE 2 - VARIABLES ORIGINALES")
print("=" * 78)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])
y = labels["Class"]



# 3. Même sous-ensemble de 100 gènes que Méthode 1


variances = X.var(axis=0)

selected_genes = (
    variances[variances > 0]
    .sort_values(ascending=False)
    .head(100)
    .index
    .tolist()
)

X_selected = X[selected_genes]

classes = sorted(y.unique())



# 4. Fonction Overlap euclidien


def overlap_euclidean(class1_data, class2_data):
    """
    Calcule l'Overlap entre deux classes
    selon la définition du TP.
    """

    center1 = class1_data.mean(axis=0)
    center2 = class2_data.mean(axis=0)

    intra1 = np.linalg.norm(
        class1_data - center1,
        axis=1
    ).max()

    intra2 = np.linalg.norm(
        class2_data - center2,
        axis=1
    ).max()

    dist_1_to_2 = np.linalg.norm(
        class1_data - center2,
        axis=1
    ).min()

    dist_2_to_1 = np.linalg.norm(
        class2_data - center1,
        axis=1
    ).min()

    inter = min(
        dist_1_to_2,
        dist_2_to_1
    )

    return (
        intra1 + intra2
    ) / (
        2 * inter
    )



# 5. Recherche gloutonne pour chaque paire de classes


results = []

for class1, class2 in combinations(classes, 2):

    print("\n" + "-" * 78)
    print(f"{class1} vs {class2}")
    print("-" * 78)

    mask1 = (y == class1)
    mask2 = (y == class2)

    
    # Étape A : meilleur premier gène
    

    first_gene_results = []

    for gene in selected_genes:

        data1 = X_selected.loc[
            mask1,
            [gene]
        ].to_numpy()

        data2 = X_selected.loc[
            mask2,
            [gene]
        ].to_numpy()

        overlap = overlap_euclidean(
            data1,
            data2
        )

        first_gene_results.append(
            (gene, overlap)
        )

    first_gene_results.sort(
        key=lambda x: x[1]
    )

    gene1 = first_gene_results[0][0]
    overlap_1d = first_gene_results[0][1]


    
    # Étape B : meilleur second gène compte tenu du premier
    

    second_gene_results = []

    for gene2 in selected_genes:

        if gene2 == gene1:
            continue

        pair = [gene1, gene2]

        data1 = X_selected.loc[
            mask1,
            pair
        ].to_numpy()

        data2 = X_selected.loc[
            mask2,
            pair
        ].to_numpy()

        overlap = overlap_euclidean(
            data1,
            data2
        )

        second_gene_results.append(
            (gene2, overlap)
        )

    second_gene_results.sort(
        key=lambda x: x[1]
    )

    gene2 = second_gene_results[0][0]
    overlap_2d = second_gene_results[0][1]


    print(
        f"Premier gène : {gene1} "
        f"(Overlap 1D = {overlap_1d:.4f})"
    )

    print(
        f"Deuxième gène : {gene2} "
        f"(Overlap 2D = {overlap_2d:.4f})"
    )


    
    # 6. Scatter plot
    

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_selected.loc[mask1, gene1],
        X_selected.loc[mask1, gene2],
        label=class1,
        alpha=0.7
    )

    plt.scatter(
        X_selected.loc[mask2, gene1],
        X_selected.loc[mask2, gene2],
        label=class2,
        alpha=0.7
    )

    plt.xlabel(gene1)
    plt.ylabel(gene2)

    plt.title(
        f"{class1} vs {class2}\n"
        f"Overlap 2D = {overlap_2d:.3f}"
    )

    plt.legend()
    plt.tight_layout()

    figure_file = (
        FIGURE_DIR
        / f"original_{class1}_{class2}.png"
    )

    plt.savefig(
        figure_file,
        dpi=300
    )

    plt.close()


    
    # 7. Sauvegarde des résultats
    

    results.append(
        {
            "classe_1": class1,
            "classe_2": class2,
            "gene_1": gene1,
            "overlap_1d": overlap_1d,
            "gene_2": gene2,
            "overlap_2d": overlap_2d,
            "bien_separees": overlap_2d < 1
        }
    )



# 8. Tableau résumé


results_df = pd.DataFrame(results)

print("\n" + "=" * 78)
print("RÉSUMÉ")
print("=" * 78)

print(
    results_df.to_string(
        index=False,
        formatters={
            "overlap_1d": "{:.4f}".format,
            "overlap_2d": "{:.4f}".format
        }
    )
)



# 9. Export


output_file = (
    TABLE_DIR
    / "original_variable_selection.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nFichier créé :")
print(output_file)

print("\nFigures créées dans :")
print(FIGURE_DIR)

print("\nMéthode 2 - variables originales terminée.")