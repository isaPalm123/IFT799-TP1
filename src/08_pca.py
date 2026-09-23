"""
IFT 799 - Science des données
TP1 - Méthode 2 : ACP

Objectif :
- utiliser le même sous-ensemble de 100 gènes que pour Mahalanobis;
- centrer les données;
- appliquer l'ACP;
- afficher les deux premières composantes principales;
- produire les résultats nécessaires pour interprétation.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
print("IFT799 - TP1 : ACP")
print("=" * 78)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])
y = labels["Class"]


# 3. Même sélection de 100 gènes

variances = X.var(axis=0)

selected_genes = (
    variances[variances > 0]
    .sort_values(ascending=False)
    .head(100)
    .index
    .tolist()
)

X_selected = X[selected_genes]

print("\nNombre de patients :", X_selected.shape[0])
print("Nombre de variables :", X_selected.shape[1])


# 4. Standardisation
#
# Le cours indique qu'on centre les variables et qu'on peut
# les standardiser lorsque nécessaire.
#
# Ici on applique StandardScaler :
# moyenne = 0
# écart-type = 1
#

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_selected)


# 5. ACP complète

pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_


print("\nVARIANCE EXPLIQUÉE")
print("-" * 78)

print(
    f"CP1 : {explained_variance[0] * 100:.2f}%"
)

print(
    f"CP2 : {explained_variance[1] * 100:.2f}%"
)

print(
    f"CP1 + CP2 : "
    f"{(explained_variance[0] + explained_variance[1]) * 100:.2f}%"
)


# 6. Variance cumulée

cumulative_variance = explained_variance.cumsum()

thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]

print("\nNOMBRE DE COMPOSANTES POUR VARIANCE CUMULÉE")
print("-" * 78)

for threshold in thresholds:

    number_components = (
        cumulative_variance >= threshold
    ).argmax() + 1

    print(
        f"{threshold * 100:.0f}% : "
        f"{number_components} composantes"
    )


# 7. Tableau des composantes

pca_results = pd.DataFrame(
    {
        "PC1": X_pca_full[:, 0],
        "PC2": X_pca_full[:, 1],
        "Class": y
    }
)

output_table = (
    TABLE_DIR
    / "pca_coordinates.csv"
)

pca_results.to_csv(
    output_table,
    index=False
)


# 8. Scatter plot global

plt.figure(figsize=(9, 7))

classes = sorted(y.unique())

for cancer_class in classes:

    mask = pca_results["Class"] == cancer_class

    plt.scatter(
        pca_results.loc[mask, "PC1"],
        pca_results.loc[mask, "PC2"],
        label=cancer_class,
        alpha=0.7
    )

plt.xlabel(
    f"CP1 ({explained_variance[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"CP2 ({explained_variance[1] * 100:.2f}% variance)"
)

plt.title(
    "ACP - Projection des cinq types de cancer"
)

plt.legend()

plt.tight_layout()

global_figure = (
    FIGURE_DIR
    / "pca_all_classes.png"
)

plt.savefig(
    global_figure,
    dpi=300
)

plt.close()


# 9. Scatter plots par paire de classes

from itertools import combinations

for class1, class2 in combinations(classes, 2):

    plt.figure(figsize=(8, 6))

    mask1 = pca_results["Class"] == class1
    mask2 = pca_results["Class"] == class2

    plt.scatter(
        pca_results.loc[mask1, "PC1"],
        pca_results.loc[mask1, "PC2"],
        label=class1,
        alpha=0.7
    )

    plt.scatter(
        pca_results.loc[mask2, "PC1"],
        pca_results.loc[mask2, "PC2"],
        label=class2,
        alpha=0.7
    )

    plt.xlabel(
        f"CP1 ({explained_variance[0] * 100:.2f}%)"
    )

    plt.ylabel(
        f"CP2 ({explained_variance[1] * 100:.2f}%)"
    )

    plt.title(
        f"ACP - {class1} vs {class2}"
    )

    plt.legend()
    plt.tight_layout()

    figure_file = (
        FIGURE_DIR
        / f"pca_{class1}_{class2}.png"
    )

    plt.savefig(
        figure_file,
        dpi=300
    )

    plt.close()


# 10. Export variance expliquée

variance_table = pd.DataFrame(
    {
        "composante": [
            f"CP{i + 1}"
            for i in range(len(explained_variance))
        ],
        "variance_expliquee": explained_variance,
        "variance_cumulee": cumulative_variance
    }
)

variance_file = (
    TABLE_DIR
    / "pca_explained_variance.csv"
)

variance_table.to_csv(
    variance_file,
    index=False
)


# 11. Résumé

print("\nFichiers créés :")

print(output_table)
print(variance_file)

print("\nFigure globale :")
print(global_figure)

print("\nFigures par paire créées dans :")
print(FIGURE_DIR)

print("\nACP terminée.")