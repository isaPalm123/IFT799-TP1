"""
IFT 799 - Science des données
TP1 - Sélection d'un sous-ensemble de variables

Objectif :
Comparer plusieurs tailles de sous-ensembles de gènes
sélectionnés selon leur variance.
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# 2. Chargement


print("=" * 70)
print("IFT799 - TP1 : SÉLECTION DES VARIABLES")
print("=" * 70)

data = pd.read_csv(DATA_FILE)

id_column = data.columns[0]

X = data.drop(columns=[id_column])



# 3. Suppression des variables constantes


variances = X.var(axis=0)

non_constant_variances = variances[variances > 0]

ranked_variances = non_constant_variances.sort_values(
    ascending=False
)



# 4. Variance totale


total_variance = ranked_variances.sum()

print("\n1. VARIANCE TOTALE")
print("-" * 70)

print(f"Nombre de variables non constantes : {len(ranked_variances)}")
print(f"Variance totale : {total_variance:.4f}")



# 5. Comparaison de plusieurs sous-ensembles


subset_sizes = [10, 25, 50, 100, 200, 500, 1000]

results = []

for size in subset_sizes:

    selected = ranked_variances.head(size)

    subset_variance = selected.sum()

    percentage = (
        subset_variance / total_variance
    ) * 100

    results.append(
        {
            "nombre_variables": size,
            "variance_cumulee": subset_variance,
            "pourcentage_variance_totale": percentage,
        }
    )


results_df = pd.DataFrame(results)

print("\n2. COMPARAISON DES SOUS-ENSEMBLES")
print("-" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "variance_cumulee": "{:.4f}".format,
            "pourcentage_variance_totale": "{:.2f}%".format,
        }
    )
)



# 6. Top 100 variables


top_100 = (
    ranked_variances
    .head(100)
    .rename("variance")
    .reset_index()
    .rename(columns={"index": "gene"})
)

output_file = OUTPUT_DIR / "top_100_genes_variance.csv"

top_100.to_csv(
    output_file,
    index=False
)


print("\n3. EXPORT")
print("-" * 70)

print(f"Top 100 sauvegardé dans : {output_file}")

print("\nAnalyse terminée.")