"""
IFT 799 - Science des données
TP1 - Méthode 2 : t-SNE

Objectif :
- utiliser le même sous-ensemble de 100 gènes;
- standardiser les données;
- appliquer t-SNE en 2D;
- visualiser les cinq classes sur une seule figure;
- sauvegarder les coordonnées pour le rapport.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE



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
print("IFT799 - TP1 : t-SNE")
print("=" * 78)

data = pd.read_csv(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])
y = labels["Class"]



# 3. Même sous-ensemble de 100 gènes


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


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_selected)



# 5. t-SNE


print("\nCalcul t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

print("t-SNE terminé.")



# 6. Tableau des coordonnées


tsne_results = pd.DataFrame(
    {
        "TSNE1": X_tsne[:, 0],
        "TSNE2": X_tsne[:, 1],
        "Class": y
    }
)

output_table = (
    TABLE_DIR
    / "tsne_coordinates.csv"
)

tsne_results.to_csv(
    output_table,
    index=False
)



# 7. Figure globale


plt.figure(figsize=(9, 7))

classes = sorted(y.unique())

for cancer_class in classes:

    mask = tsne_results["Class"] == cancer_class

    plt.scatter(
        tsne_results.loc[mask, "TSNE1"],
        tsne_results.loc[mask, "TSNE2"],
        label=cancer_class,
        alpha=0.7
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.title(
    "t-SNE - Projection des cinq types de cancer"
)

plt.legend()
plt.tight_layout()

figure_file = (
    FIGURE_DIR
    / "tsne_all_classes.png"
)

plt.savefig(
    figure_file,
    dpi=300
)

plt.close()



# 8. Résumé


print("\nFichier créé :")
print(output_table)

print("\nFigure créée :")
print(figure_file)

print("\nt-SNE terminé.")