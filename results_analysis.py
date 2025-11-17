import pandas as pd
import matplotlib.pyplot as plt

number_of_molecules = 11

def process_predictions(prediction_file, resistant_inchikeys, model_name):
    """Loads predictions, filters out resistant molecules, and ranks them."""
    df = pd.read_csv(prediction_file)
    print(f"\n--- Processing {model_name} ---")
    print(f"Number of molecules in predictions file: {len(df)}")
    
    filtered_df = df[~df['InChIKey'].isin(resistant_inchikeys)]
    print(f"Number of molecules after filtering resistant set: {len(filtered_df)}")

    ordered_df = filtered_df.sort_values(by='prediction', ascending=False)
    ordered_df = ordered_df.reset_index(drop=True)
    ordered_df['rank'] = ordered_df.index
    return ordered_df


def _top_sets_by_percent(ranks_df, percent):
    """Return top-k sets for each model given a percent threshold."""
    total = len(ranks_df)
    k = max(1, int((percent / 100.0) * total))
    mol_top = set(ranks_df.nsmallest(k, 'rank_molformer')['InChIKey'])
    che_top = set(ranks_df.nsmallest(k, 'rank_chemprop')['InChIKey'])
    rf_top = set(ranks_df.nsmallest(k, 'rank_rf')['InChIKey'])
    return k, mol_top, che_top, rf_top


def compute_overlap_curves(ranks_df, percents):
    """Compute pairwise and triple overlaps (as percentages) across top n%.

    Returns dict with keys for each curve containing lists aligned to percents.
    """
    curves = {
        'Molformer ∩ Chemprop': [],
        'Molformer ∩ RandomForest': [],
        'Chemprop ∩ RandomForest': [],
        'All three': [],
    }

    for p in percents:
        k, mol_top, che_top, rf_top = _top_sets_by_percent(ranks_df, p)
        denom = float(k)
        mol_che = (len(mol_top & che_top) / denom) * 100.0
        mol_rf = (len(mol_top & rf_top) / denom) * 100.0
        che_rf = (len(che_top & rf_top) / denom) * 100.0
        all_three = (len(mol_top & che_top & rf_top) / denom) * 100.0

        curves['Molformer ∩ Chemprop'].append(mol_che)
        curves['Molformer ∩ RandomForest'].append(mol_rf)
        curves['Chemprop ∩ RandomForest'].append(che_rf)
        curves['All three'].append(all_three)

    return curves


def plot_overlap_curves(percents, curves, output_path, confidence_intervals=None):
    """Plot overlap percentage curves vs top n% and save the figure."""
    plt.figure(figsize=(7, 5))
    for label, values in curves.items():
        if confidence_intervals is not None and label in confidence_intervals:
            ci_low, ci_high = confidence_intervals[label]
            plt.fill_between(percents, ci_low, ci_high, alpha=0.15)
        plt.plot(percents, values, label=label, linewidth=2)
    plt.xlabel('Top n% of ranked molecules')
    plt.ylabel('Percentage overlap')
    plt.title('Overlap of top-ranked molecules across models')
    plt.legend()
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_unique_curves(ranks_df, percents):
    """Compute percentage of unique molecules in top n% for each model."""
    curves = {
        'Molformer unique': [],
        'Chemprop unique': [],
        'RandomForest unique': [],
    }

    for p in percents:
        k, mol_top, che_top, rf_top = _top_sets_by_percent(ranks_df, p)
        denom = float(k)
        mol_unique = (len(mol_top - (che_top | rf_top)) / denom) * 100.0
        che_unique = (len(che_top - (mol_top | rf_top)) / denom) * 100.0
        rf_unique = (len(rf_top - (mol_top | che_top)) / denom) * 100.0

        curves['Molformer unique'].append(mol_unique)
        curves['Chemprop unique'].append(che_unique)
        curves['RandomForest unique'].append(rf_unique)

    return curves


def plot_unique_curves(percents, curves, output_path, confidence_intervals=None):
    """Plot unique percentage curves vs top n% and save the figure.

    confidence_intervals: optional dict mapping label -> (lower_list, upper_list)
    """
    plt.figure(figsize=(7, 5))
    for label, values in curves.items():
        if confidence_intervals is not None and label in confidence_intervals:
            ci_low, ci_high = confidence_intervals[label]
            plt.fill_between(percents, ci_low, ci_high, alpha=0.15)
        plt.plot(percents, values, label=label, linewidth=2)
    plt.xlabel('Top n% of ranked molecules')
    plt.ylabel('Percentage unique')
    plt.title('Unique molecules in top-ranked sets per model')
    plt.legend()
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_k_values(total_count, percents):
    """Compute top-k size per percent for the common ranked set."""
    return [max(1, int((p / 100.0) * total_count)) for p in percents]


def compute_binomial_confidence_intervals(unique_curves, k_values, z_value=1.96):
    """Compute 95% CIs using binomial SE for each uniqueness curve.

    Returns dict: label -> (lower_list, upper_list) in percent units.
    """
    ci = {}
    for label, values in unique_curves.items():
        lower, upper = [], []
        for i, k in enumerate(k_values):
            p_hat = min(1.0, max(0.0, values[i] / 100.0))
            se = (p_hat * (1.0 - p_hat) / float(k)) ** 0.5
            lo = max(0.0, (p_hat - z_value * se) * 100.0)
            hi = min(100.0, (p_hat + z_value * se) * 100.0)
            lower.append(lo)
            upper.append(hi)
        ci[label] = (lower, upper)
    return ci

# Read resistant dataset and generate InChIKeys for filtering
resistant_df = pd.read_csv('processed_datasets/10uM_FP_clustered_resistant_pneumococcus_augmented_dataset.csv')
resistant_df = resistant_df[resistant_df['source']!='MRSA_Wong2024']
resistant_inchikeys = set(resistant_df['InChIKey'].dropna())

# Process predictions for each model
molformer_df = process_predictions('results/molformer_predictions.csv', resistant_inchikeys, "Molformer")
chemprop_df = process_predictions('results/chemprop_predictions.csv', resistant_inchikeys, "Chemprop")
rf_df = process_predictions('results/rf_predictions.csv', resistant_inchikeys, "Random Forest")

# --- Spearman Correlation of Ranks Across Methods ---
ranks_df = (
    molformer_df[['InChIKey', 'rank']]
    .merge(chemprop_df[['InChIKey', 'rank']], on='InChIKey', suffixes=('_molformer', '_chemprop'))
    .merge(rf_df[['InChIKey', 'rank']], on='InChIKey')
)
ranks_df.rename(columns={'rank': 'rank_rf'}, inplace=True)
rank_corr = ranks_df[['rank_molformer', 'rank_chemprop', 'rank_rf']].corr(method='spearman')
print("\n--- Spearman correlation of ranks across methods ---")
print(rank_corr)

# --- Spearman Correlation for Top 10% by Average Rank (across 3 methods) ---
avg_rank_colnames = ['rank_molformer', 'rank_chemprop', 'rank_rf']
ranks_df['avg_rank'] = ranks_df[avg_rank_colnames].mean(axis=1)
top_n = max(1, int(0.10 * len(ranks_df)))
top_subset = ranks_df.nsmallest(top_n, 'avg_rank')
top_rank_corr = top_subset[avg_rank_colnames].corr(method='spearman')
print("\n--- Spearman correlation of ranks for top 10% by average rank ---")
print(top_rank_corr)

# --- Top 10% Overlap Between Models (on common molecule set) ---
top_k_10 = max(1, int(0.10 * len(ranks_df)))
mol_top_set = set(ranks_df.nsmallest(top_k_10, 'rank_molformer')['InChIKey'])
che_top_set = set(ranks_df.nsmallest(top_k_10, 'rank_chemprop')['InChIKey'])
rf_top_set = set(ranks_df.nsmallest(top_k_10, 'rank_rf')['InChIKey'])

mol_che_overlap = len(mol_top_set & che_top_set)
mol_rf_overlap = len(mol_top_set & rf_top_set)
che_rf_overlap = len(che_top_set & rf_top_set)
all_three_overlap = len(mol_top_set & che_top_set & rf_top_set)

den = float(top_k_10)
print("\n--- Top 10% overlap of ranks (common molecules) ---")
print(f"Molformer ∩ Chemprop: {mol_che_overlap} of {top_k_10} ({(mol_che_overlap/den)*100:.1f}%)")
print(f"Molformer ∩ RandomForest: {mol_rf_overlap} of {top_k_10} ({(mol_rf_overlap/den)*100:.1f}%)")
print(f"Chemprop ∩ RandomForest: {che_rf_overlap} of {top_k_10} ({(che_rf_overlap/den)*100:.1f}%)")
print(f"All three: {all_three_overlap} of {top_k_10} ({(all_three_overlap/den)*100:.1f}%)")

# --- Overlap vs Top n% Graph ---
percents = list(range(100, 0, -1))  # 100% down to 1%
overlap_curves = compute_overlap_curves(ranks_df, percents)
k_values = compute_k_values(len(ranks_df), percents)
overlap_ci = compute_binomial_confidence_intervals(overlap_curves, k_values)
plot_overlap_curves(percents, overlap_curves, 'results/overlap_vs_top_percent.png', confidence_intervals=overlap_ci)

# --- Unique-in-top n% Graph (per model) ---
unique_curves = compute_unique_curves(ranks_df, percents)
unique_ci = compute_binomial_confidence_intervals(unique_curves, k_values)
plot_unique_curves(percents, unique_curves, 'results/unique_in_top_percent.png', confidence_intervals=unique_ci)

# --- Top N Predictions for Each Model ---
print(f"\n--- Top {number_of_molecules} Predictions ---")

print("\nMolformer:")
print(molformer_df.head(number_of_molecules)[['vendor_name', 'pert_iname', 'prediction', 'uncertainty']])

print("\nChemprop:")
chemprop_cols = ['vendor_name', 'pert_iname', 'prediction']
if 'uncertainty' in chemprop_df.columns:
    chemprop_cols.append('uncertainty')
print(chemprop_df.head(number_of_molecules)[chemprop_cols])

print("\nRandom Forest:")
rf_cols = ['vendor_name', 'pert_iname', 'prediction']
if 'uncertainty' in rf_df.columns:
    rf_cols.append('uncertainty')
print(rf_df.head(number_of_molecules)[rf_cols])

# --- Top N for Averaged Molformer and Chemprop ---
print(f"\n--- Top {number_of_molecules} for Molformer and Chemprop Average ---")
merged_df = pd.merge(molformer_df[['InChIKey', 'vendor_name', 'pert_iname', 'prediction']],
                     chemprop_df[['InChIKey', 'prediction']],
                     on='InChIKey',
                     suffixes=('_molformer', '_chemprop'))

merged_df['average_prediction'] = (merged_df['prediction_molformer'] + merged_df['prediction_chemprop']) / 2
ensemble_df = merged_df.sort_values(by='average_prediction', ascending=False).reset_index(drop=True)
ensemble_df['rank'] = ensemble_df.index
print(ensemble_df.head(number_of_molecules)[['vendor_name', 'pert_iname', 'average_prediction', 'rank']])


# --- Averaged All Three ---
merged_all_df = pd.merge(merged_df,
                         rf_df[['InChIKey', 'prediction']],
                         on='InChIKey')
merged_all_df.rename(columns={'prediction': 'prediction_rf'}, inplace=True)
merged_all_df['average_prediction_all'] = (merged_all_df['prediction_molformer'] + merged_all_df['prediction_chemprop'] + merged_all_df['prediction_rf']) / 3
ensemble_all_df = merged_all_df.sort_values(by='average_prediction_all', ascending=False).reset_index(drop=True)
ensemble_all_df['rank'] = ensemble_all_df.index


# --- Predictions for Specific Drugs ---
drugs = ['dactinomycin', 'Cadazolid', 'Cefazolin (sodium)', 'Cefodizime', 'Cefotetan', 
         'cefotiam-cilexetil', 'Cefozopran', 'Ceftiofur', 'Levofloxacin', 'Linaclotide', 'Thiostrepton']

print("\n--- Predictions for Specific Drugs ---")
for drug in drugs:
    print(f"\n{drug}:")

    # Ensemble (Molformer + Chemprop Average)
    ens_res = ensemble_df[(ensemble_df['vendor_name'].str.lower() == drug.lower()) | (ensemble_df['pert_iname'].str.lower() == drug.lower())]
    if not ens_res.empty:
        print(f"  Ensemble:      Average Prediction={ens_res['average_prediction'].values[0]:.3f}, Rank={ens_res['rank'].values[0]+1}")
    else:
        print("  Ensemble:      Not found")
        
    # Molformer
    mol_res = molformer_df[(molformer_df['vendor_name'].str.lower() == drug.lower()) | (molformer_df['pert_iname'].str.lower() == drug.lower())]
    if not mol_res.empty:
        print(f"  Molformer:     Prediction={mol_res['prediction'].values[0]:.3f}, Rank={mol_res['rank'].values[0]+1}, Uncertainty={mol_res['uncertainty'].values[0]:.3f}")
    else:
        print("  Molformer:     Not found")

    # Chemprop
    che_res = chemprop_df[(chemprop_df['vendor_name'].str.lower() == drug.lower()) | (chemprop_df['pert_iname'].str.lower() == drug.lower())]
    if not che_res.empty:
        rank_info = f"Rank={che_res['rank'].values[0]+1}"
        uncertainty_info = f", Uncertainty={che_res['uncertainty'].values[0]:.3f}" if 'uncertainty' in che_res.columns else ""
        print(f"  Chemprop:      Prediction={che_res['prediction'].values[0]:.3f}, {rank_info}{uncertainty_info}")
    else:
        print("  Chemprop:      Not found")

    # Random Forest
    rf_res = rf_df[(rf_df['vendor_name'].str.lower() == drug.lower()) | (rf_df['pert_iname'].str.lower() == drug.lower())]
    if not rf_res.empty:
        rank_info = f"Rank={rf_res['rank'].values[0]+1}"
        uncertainty_info = f", Uncertainty={rf_res['uncertainty'].values[0]:.3f}" if 'uncertainty' in rf_res.columns else ""
        print(f"  Random Forest: Prediction={rf_res['prediction'].values[0]:.3f}, {rank_info}{uncertainty_info}")
    else:
        print("  Random Forest: Not found")
    