"""
IFT 799 - Science des données
TP1 - Inspection des résultats par paire de classes

Objectif :
Comparer les résultats détaillés des sous-ensembles
de 100 et 200 variables.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETAIL_FILE = (
    PROJECT_ROOT
    / "output"
    / "tables"
    / "feature_subset_pairwise_details.csv"
)



# Chargement


df = pd.read_csv(DETAIL_FILE)



# Top 100


top100 = (
    df[df["nombre_variables"] == 100]
    .sort_values("overlap")
)


print("=" * 75)
print("TOP 100 GÈNES")
print("=" * 75)

print(
    top100[
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
            "dist_intra_1": "{:.4f}".format,
            "dist_intra_2": "{:.4f}".format,
            "dist_inter": "{:.4f}".format,
            "overlap": "{:.4f}".format,
        }
    )
)



# Top 200


top200 = (
    df[df["nombre_variables"] == 200]
    .sort_values("overlap")
)


print("\n" + "=" * 75)
print("TOP 200 GÈNES")
print("=" * 75)

print(
    top200[
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
            "dist_intra_1": "{:.4f}".format,
            "dist_intra_2": "{:.4f}".format,
            "dist_inter": "{:.4f}".format,
            "overlap": "{:.4f}".format,
        }
    )
)