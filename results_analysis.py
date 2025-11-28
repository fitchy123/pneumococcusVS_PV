import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

number_of_molecules = 20

def process_predictions(prediction_file, resistant_inchikeys, model_name, resistant_smiles=None):
    """Loads predictions, filters out resistant molecules, and ranks them."""
    df = pd.read_csv(prediction_file)
    print(f"\n--- Processing {model_name} ---")
    print(f"Number of molecules in predictions file: {len(df)}")
    
    filtered_df = df[~df['InChIKey'].isin(resistant_inchikeys)]
    print(f"Number of molecules after filtering resistant set: {len(filtered_df)}")
    if resistant_smiles is not None:
        smiles_filtered_df = df[~df['smiles'].isin(resistant_smiles)]
        print(f"Number of molecules after filtering resistant smiles: {len(smiles_filtered_df)}")

    ordered_df = filtered_df.sort_values(by='prediction', ascending=False)
    ordered_df = ordered_df.reset_index(drop=True)
    ordered_df['rank'] = ordered_df.index + 1
    return ordered_df


def _top_sets_by_percent(ranks_df, percent):
    """Return top-k sets for each model given a percent threshold."""
    total = len(ranks_df)
    k = max(1, int((percent / 100.0) * total))
    mol_top = set(ranks_df.nsmallest(k, 'rank_molformer')['smiles'])
    che_top = set(ranks_df.nsmallest(k, 'rank_chemprop')['smiles'])
    rf_top = set(ranks_df.nsmallest(k, 'rank_rf')['smiles'])
    return k, mol_top, che_top, rf_top


def compute_overlap_curves(ranks_df, percents):
    """Compute pairwise and triple overlaps (as percentages) across top n%.

    Returns dict with keys for each curve containing lists aligned to percents.
    """
    curves = {
        'MoLFormer ensemble ∩ D-MPNN ensemble': [],
        'MoLFormer ensemble ∩ RF': [],
        'D-MPNN ensemble ∩ RF': [],
        'All three': [],
    }

    for p in percents:
        k, mol_top, che_top, rf_top = _top_sets_by_percent(ranks_df, p)
        denom = float(k)
        mol_che = (len(mol_top & che_top) / denom) * 100.0
        mol_rf = (len(mol_top & rf_top) / denom) * 100.0
        che_rf = (len(che_top & rf_top) / denom) * 100.0
        all_three = (len(mol_top & che_top & rf_top) / denom) * 100.0

        curves['MoLFormer ensemble ∩ D-MPNN ensemble'].append(mol_che)
        curves['MoLFormer ensemble ∩ RF'].append(mol_rf)
        curves['D-MPNN ensemble ∩ RF'].append(che_rf)
        curves['All three'].append(all_three)

    return curves


def _plot_overlap_on_axis(ax, percents, curves, confidence_intervals=None, use_dashes=False, panel_label=None):
    """Helper to plot overlap curves on a given axis."""
    linestyles = ['-', '--', '-.', ':']
    for idx, (label, values) in enumerate(curves.items()):
        if confidence_intervals is not None and label in confidence_intervals:
            ci_low, ci_high = confidence_intervals[label]
            ax.fill_between(percents, ci_low, ci_high, alpha=0.15)
        linestyle = linestyles[idx % len(linestyles)] if use_dashes else '-'
        ax.plot(percents, values, label=label, linewidth=2, linestyle=linestyle)
    ax.set_xlabel('Top n% of ranked molecules', fontsize=15)
    ax.set_ylabel('Percentage overlap', fontsize=15)
    ax.set_title('Overlap of top-ranked molecules across models', fontsize=15)
    ax.legend(fontsize=10.5)
    ax.tick_params(axis='both', labelsize=10.5)
    ax.invert_xaxis()
    if panel_label:
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

def plot_overlap_curves(percents, curves, output_path, confidence_intervals=None, use_dashes=False):
    """Plot overlap percentage curves vs top n% and save the figure.
    
    use_dashes: If True, use different line styles for each curve. If False, use solid lines (default).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_overlap_on_axis(ax, percents, curves, confidence_intervals, use_dashes)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()


def compute_unique_curves(ranks_df, percents):
    """Compute percentage of unique molecules in top n% for each model."""
    curves = {
        'MoLFormer ensemble unique': [],
        'D-MPNN ensemble unique': [],
        'RF unique': [],
    }

    for p in percents:
        k, mol_top, che_top, rf_top = _top_sets_by_percent(ranks_df, p)
        denom = float(k)
        mol_unique = (len(mol_top - (che_top | rf_top)) / denom) * 100.0
        che_unique = (len(che_top - (mol_top | rf_top)) / denom) * 100.0
        rf_unique = (len(rf_top - (mol_top | che_top)) / denom) * 100.0

        curves['MoLFormer ensemble unique'].append(mol_unique)
        curves['D-MPNN ensemble unique'].append(che_unique)
        curves['RF unique'].append(rf_unique)

    return curves


def _plot_unique_on_axis(ax, percents, curves, confidence_intervals=None, use_dashes=False, panel_label=None):
    """Helper to plot unique curves on a given axis."""
    linestyles = ['-', '--', '-.']
    for idx, (label, values) in enumerate(curves.items()):
        if confidence_intervals is not None and label in confidence_intervals:
            ci_low, ci_high = confidence_intervals[label]
            ax.fill_between(percents, ci_low, ci_high, alpha=0.15)
        linestyle = linestyles[idx % len(linestyles)] if use_dashes else '-'
        ax.plot(percents, values, label=label, linewidth=2, linestyle=linestyle)
    ax.set_xlabel('Top n% of ranked molecules', fontsize=15)
    ax.set_ylabel('Percentage unique', fontsize=15)
    ax.set_title('Unique molecules in top-ranked sets per model', fontsize=15)
    ax.legend(fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.invert_xaxis()
    if panel_label:
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

def plot_unique_curves(percents, curves, output_path, confidence_intervals=None, use_dashes=False):
    """Plot unique percentage curves vs top n% and save the figure.

    confidence_intervals: optional dict mapping label -> (lower_list, upper_list)
    use_dashes: If True, use different line styles for each curve. If False, use solid lines (default).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_unique_on_axis(ax, percents, curves, confidence_intervals, use_dashes)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()


def plot_overlap_and_unique_combined(percents, overlap_curves, unique_curves, output_path, 
                                      overlap_ci=None, unique_ci=None, use_dashes=False):
    """Plot overlap and unique curves side by side in a single figure, labeled b and c."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_overlap_on_axis(ax1, percents, overlap_curves, overlap_ci, use_dashes, panel_label='a')
    _plot_unique_on_axis(ax2, percents, unique_curves, unique_ci, use_dashes, panel_label='b')
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
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


def collect_prediction_series(molformer_df, chemprop_df, rf_df, merged_df):
    """Collect prediction Series per model and MoLFormer-D-MPNN average, excluding -1s."""
    mol_series = molformer_df.loc[molformer_df['prediction'] >= 0, 'prediction']
    che_series = chemprop_df.loc[chemprop_df['prediction'] >= 0, 'prediction']
    rf_series = rf_df.loc[rf_df['prediction'] >= 0, 'prediction']
    valid_avg_mask = (merged_df['prediction_molformer'] >= 0) & (merged_df['prediction_chemprop'] >= 0)
    avg_series = ((merged_df.loc[valid_avg_mask, 'prediction_molformer'] + merged_df.loc[valid_avg_mask, 'prediction_chemprop']) / 2)
    return {
        'MoLFormer ensemble': mol_series,
        'D-MPNN ensemble': che_series,
        'RF': rf_series,
        'Consensus of MoLFormer and D-MPNN ensembles': avg_series,
    }


def plot_prediction_histograms_grid(pred_series, output_path, bins=50, xlim=(0.0, 1.0), log_x=False, log_y=False, density=True, transform=None, same_y_scale=False, kde=False, kde_bw_adjust=1.0, panel_label=None):
    """Plot 2x2 grid of histograms for prediction distributions.

    transform: None or 'log10'. If 'log10', data are transformed first and binned linearly.
    density: If False, plot counts instead of density.
    same_y_scale: If True, force identical y-limits across all subplots.
    kde: If True, overlay seaborn KDE curve (always as density).
    kde_bw_adjust: Bandwidth adjustment for KDE (larger = smoother).
    panel_label: Optional label (e.g., 'a', 'b') to add to top-left of figure.
    """
    fontsize = 24
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    axes = axes.flatten()
    labels = list(pred_series.keys())
    # Wrap long titles to prevent cut-off
    wrapped_labels = []
    for label in labels:
        if len(label) > 35:
            # Split long titles at logical break points
            if 'Consensus' in label:
                wrapped_labels.append('Consensus of MoLFormer \nand D-MPNN ensembles')
            else:
                wrapped_labels.append(label)
        else:
            wrapped_labels.append(label)
    
    for idx, (label, wrapped_label) in enumerate(zip(labels, wrapped_labels)):
        data = pred_series[label].dropna().values
        if transform == 'log10':
            data = data[data > 0]
            if data.size > 0:
                lower = xlim[0] if xlim is not None else max(data.min(), 1e-12)
                lower = max(lower, 1e-12)
                upper = xlim[1] if xlim is not None else data.max()
                bin_edges = np.linspace(np.log10(lower), np.log10(upper), bins + 1)
                axes[idx].hist(np.log10(data), bins=bin_edges, density=density, alpha=0.7, log=log_y)
                if kde:
                    sns.kdeplot(x=np.log10(data), ax=axes[idx], bw_adjust=kde_bw_adjust, color=None, linewidth=2, label=None)
                if xlim is not None:
                    axes[idx].set_xlim((np.log10(lower), np.log10(upper)))
            else:
                axes[idx].hist([], bins=bins, density=density, alpha=0.7, log=log_y)
            axes[idx].set_xlabel('log10 Predicted probability', fontsize=fontsize)
        elif log_x:
            data = data[data > 0]
            if data.size > 0:
                lower = xlim[0] if xlim is not None else max(data.min(), 1e-12)
                lower = max(lower, 1e-12)
                upper = xlim[1] if xlim is not None else data.max()
                bin_edges = np.logspace(np.log10(lower), np.log10(upper), bins + 1)
                axes[idx].hist(data, bins=bin_edges, density=density, alpha=0.7, log=log_y)
                if kde:
                    sns.kdeplot(x=data, ax=axes[idx], bw_adjust=kde_bw_adjust, color=None, linewidth=2, label=None)
            else:
                axes[idx].hist([], bins=bins, density=density, alpha=0.7, log=log_y)
            axes[idx].set_xscale('log')
        else:
            axes[idx].hist(data, bins=bins, density=density, alpha=0.7, log=log_y)
            if kde and data.size > 0:
                sns.kdeplot(x=data, ax=axes[idx], bw_adjust=kde_bw_adjust, color=None, linewidth=2, label=None)
        axes[idx].set_title(wrapped_label, fontsize=fontsize)
        if transform != 'log10':
            axes[idx].set_xlabel('Predicted probability', fontsize=fontsize)
        if density:
            axes[idx].set_ylabel('Density', fontsize=fontsize)
        else:
            axes[idx].set_ylabel('Count', fontsize=fontsize)
        axes[idx].tick_params(axis='both', labelsize=fontsize - 2)
        if xlim is not None:
            axes[idx].set_xlim(xlim)
        if log_y:
            axes[idx].set_yscale('log')
            axes[idx].yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: ('{:.6f}'.format(y)).rstrip('0').rstrip('.'))
            )
    if same_y_scale:
        y_mins = []
        y_maxs = []
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            y_mins.append(ymin)
            y_maxs.append(ymax)
        if log_y:
            min_y = max(min(y_mins), 1e-12)
        else:
            min_y = 0.0
        max_y = max(y_maxs)
        for ax in axes:
            ax.set_ylim(min_y, max_y)
    if panel_label:
        fig.text(0.02, 0.98, panel_label, fontsize=35, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()


def plot_prediction_histograms_overlay(pred_series, output_path, bins=50, xlim=(0.0, 1.0), log_x=False, log_y=False, density=True, transform=None, kde=False, kde_bw_adjust=1.0, auto_tight_y=True, tight_y_quantile=0.02):
    """Plot overlayed KDE curves for prediction distributions (no histogram).

    transform: None or 'log10'. If 'log10', data are transformed first and binned linearly.
    density: If False, plot counts instead of density.
    kde: If True, overlay seaborn KDE curves (always as density).
    kde_bw_adjust: Bandwidth adjustment for KDE (larger = smoother).
    auto_tight_y: If True and log_y, set y-limits to tightly bracket KDE curves.
    tight_y_quantile: Lower quantile used to ignore tiny tail densities when tightening y.
    """
    plt.figure(figsize=(7, 5))
    for label, series in pred_series.items():
        data = series.dropna().values
        if transform == 'log10':
            data = data[data > 0]
            if data.size == 0:
                continue
            lower = xlim[0] if xlim is not None else max(data.min(), 1e-12)
            lower = max(lower, 1e-12)
            upper = xlim[1] if xlim is not None else data.max()
            if kde:
                sns.kdeplot(x=np.log10(data), bw_adjust=kde_bw_adjust, color=None, linewidth=2, label=f'{label} KDE')
        elif log_x:
            data = data[data > 0]
            if data.size == 0:
                continue
            lower = xlim[0] if xlim is not None else max(data.min(), 1e-12)
            lower = max(lower, 1e-12)
            upper = xlim[1] if xlim is not None else data.max()
            if kde:
                sns.kdeplot(x=data, bw_adjust=kde_bw_adjust, color=None, linewidth=2, label=f'{label} KDE')
        else:
            if kde and data.size > 0:
                sns.kdeplot(x=data, bw_adjust=kde_bw_adjust, color=None, linewidth=2, label=f'{label} KDE')
    if transform == 'log10':
        plt.xlabel('log10 Predicted probability')
    else:
        plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.title('Distribution of predicted probabilities')
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: ('{:.6f}'.format(y)).rstrip('0').rstrip('.'))
        )
    # Auto-tighten Y limits based on KDE lines (for both linear and log y)
    if auto_tight_y:
        ax = plt.gca()
        y_min = None
        y_max = None
        y_all = []
        for line in ax.get_lines():
            ydata = line.get_ydata(orig=False)
            if ydata is None:
                continue
            yvals = np.asarray(ydata)
            yvals = yvals[np.isfinite(yvals)]
            if log_y:
                yvals = yvals[yvals > 0]
            if yvals.size == 0:
                continue
            y_all.append(yvals)
        if len(y_all) > 0:
            y_all = np.concatenate(y_all)
            if log_y:
                # Ignore tiny tail values by using a quantile-based lower bound
                q = min(max(tight_y_quantile, 0.0), 0.5)
                y_min = float(np.quantile(y_all, q))
            else:
                y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))
        if y_min is not None and y_max is not None and y_max > 0:
            if log_y:
                lower = max(y_min / 1.25, 1e-12)
                upper = y_max * 1.1
            else:
                span = max(y_max - y_min, 1e-12)
                lower = max(0.0, y_min - 0.05 * span)
                upper = y_max + 0.05 * span
            ax.set_ylim(lower, upper)
    plt.legend()
    if xlim is not None:
        if transform == 'log10':
            lower = max(xlim[0], 1e-12)
            plt.xlim((np.log10(lower), np.log10(xlim[1])))
        else:
            plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()

# Read resistant dataset and generate InChIKeys for filtering
resistant_df = pd.read_csv('processed_datasets/10uM_FP_clustered_resistant_pneumococcus_augmented_dataset.csv')
resistant_df = resistant_df[resistant_df['source']!='MRSA_Wong2024']
resistant_inchikeys = set(resistant_df['InChIKey'].dropna())
#resistant_smiles = set(resistant_df['Smiles'].dropna())
# Process predictions for each model
molformer_df = process_predictions('results/molformer_predictions.csv', resistant_inchikeys, "MoLFormer")
chemprop_df = process_predictions('results/chemprop_predictions.csv', resistant_inchikeys, "D-MPNN")
rf_df = process_predictions('results/rf_predictions.csv', resistant_inchikeys, "Random Forest")

# --- Spearman Correlation of Ranks Across Methods ---
ranks_df = (
    molformer_df[['smiles', 'rank']]
    .merge(chemprop_df[['smiles', 'rank']], on='smiles', suffixes=('_molformer', '_chemprop'))
    .merge(rf_df[['smiles', 'rank']], on='smiles')
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
mol_top_set = set(ranks_df.nsmallest(top_k_10, 'rank_molformer')['smiles'])
che_top_set = set(ranks_df.nsmallest(top_k_10, 'rank_chemprop')['smiles'])
rf_top_set = set(ranks_df.nsmallest(top_k_10, 'rank_rf')['smiles'])

mol_che_overlap = len(mol_top_set & che_top_set)
mol_rf_overlap = len(mol_top_set & rf_top_set)
che_rf_overlap = len(che_top_set & rf_top_set)
all_three_overlap = len(mol_top_set & che_top_set & rf_top_set)

den = float(top_k_10)
print("\n--- Top 10% overlap of ranks (common molecules) ---")
print(f"MoLFormer ∩ D-MPNN: {mol_che_overlap} of {top_k_10} ({(mol_che_overlap/den)*100:.1f}%)")
print(f"MoLFormer ∩ RandomForest: {mol_rf_overlap} of {top_k_10} ({(mol_rf_overlap/den)*100:.1f}%)")
print(f"D-MPNN ∩ RandomForest: {che_rf_overlap} of {top_k_10} ({(che_rf_overlap/den)*100:.1f}%)")
print(f"All three: {all_three_overlap} of {top_k_10} ({(all_three_overlap/den)*100:.1f}%)")

# --- Overlap vs Top n% Graph ---
percents = list(np.arange(20, 0, -0.2))  # 100% down to 1%
overlap_curves = compute_overlap_curves(ranks_df, percents)
k_values = compute_k_values(len(ranks_df), percents)
overlap_ci = compute_binomial_confidence_intervals(overlap_curves, k_values)

# --- Unique-in-top n% Graph (per model) ---
unique_curves = compute_unique_curves(ranks_df, percents)
unique_ci = compute_binomial_confidence_intervals(unique_curves, k_values)

# Combined plot with labels b and c
plot_overlap_and_unique_combined(percents, overlap_curves, unique_curves, 
                                 'results/overlap_and_unique_combined.png',
                                 overlap_ci=overlap_ci, unique_ci=unique_ci, use_dashes=True)

# Individual plots (kept for backward compatibility)
plot_overlap_curves(percents, overlap_curves, 'results/overlap_vs_top_percent.png', confidence_intervals=overlap_ci, use_dashes=True)
plot_unique_curves(percents, unique_curves, 'results/unique_in_top_percent.png', confidence_intervals=unique_ci, use_dashes=True)

# --- Add cross-model ranks ---
molformer_df['rank_chemprop'] = [chemprop_df[chemprop_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in molformer_df['InChIKey']]
molformer_df['rank_rf'] = [rf_df[rf_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in molformer_df['InChIKey']]

chemprop_df['rank_molformer'] = [molformer_df[molformer_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in chemprop_df['InChIKey']]
chemprop_df['rank_rf'] = [rf_df[rf_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in chemprop_df['InChIKey']]

rf_df['rank_molformer'] = [molformer_df[molformer_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in rf_df['InChIKey']]
rf_df['rank_chemprop'] = [chemprop_df[chemprop_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in rf_df['InChIKey']]

# --- Top N Predictions for Each Model ---
print(f"\n--- Top {number_of_molecules} Predictions ---")

print("\nMoLFormer:")
print(molformer_df.head(number_of_molecules)[['vendor_name', 'pert_iname', 'prediction', 'rank', 'rank_chemprop', 'rank_rf']])

print("\nD-MPNN:")
print(chemprop_df.head(number_of_molecules)[['vendor_name', 'pert_iname', 'prediction', 'rank', 'rank_molformer', 'rank_rf']])

print("\nRandom Forest:")
print(rf_df.head(number_of_molecules)[['vendor_name', 'pert_iname', 'prediction', 'rank', 'rank_molformer', 'rank_chemprop']])

# --- Top N for Averaged MoLFormer and D-MPNN ---
print(f"\n--- Top {number_of_molecules} for MoLFormer and D-MPNN Average ---")
merged_df = pd.merge(molformer_df[['InChIKey', 'vendor_name', 'pert_iname', 'prediction', 'smiles']],
                     chemprop_df[['InChIKey', 'prediction']],
                     on='InChIKey',
                     suffixes=('_molformer', '_chemprop'))

merged_df['average_prediction'] = (merged_df['prediction_molformer'] + merged_df['prediction_chemprop']) / 2
ensemble_df = merged_df.sort_values(by='average_prediction', ascending=False).reset_index(drop=True)
ensemble_df['rank'] = ensemble_df.index + 1
ensemble_df['rank_molformer'] = [molformer_df[molformer_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in ensemble_df['InChIKey']]
ensemble_df['rank_chemprop'] = [chemprop_df[chemprop_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in ensemble_df['InChIKey']]
ensemble_df['rank_rf'] = [rf_df[rf_df['InChIKey'] == InChIKey]['rank'].values[0] for InChIKey in ensemble_df['InChIKey']]
print(ensemble_df.head(number_of_molecules)[['vendor_name', 'pert_iname', 'average_prediction', 'rank', 'rank_molformer', 'rank_chemprop', 'rank_rf']])

####
"""from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
top_ensemble_df = ensemble_df.head(number_of_molecules)"""

# --- Averaged All Three ---
merged_all_df = pd.merge(merged_df,
                         rf_df[['InChIKey', 'prediction']],
                         on='InChIKey')
merged_all_df.rename(columns={'prediction': 'prediction_rf'}, inplace=True)
merged_all_df['average_prediction_all'] = (merged_all_df['prediction_molformer'] + merged_all_df['prediction_chemprop'] + merged_all_df['prediction_rf']) / 3
ensemble_all_df = merged_all_df.sort_values(by='average_prediction_all', ascending=False).reset_index(drop=True)
ensemble_all_df['rank'] = ensemble_all_df.index


# --- Prediction probability distributions ---
pred_series = collect_prediction_series(molformer_df, chemprop_df, rf_df, merged_df)
plot_prediction_histograms_grid(pred_series, 'results/prediction_distributions_grid_logx.png', bins=30, xlim=(0.0, 1.0), log_x=False, log_y=True, density=False, same_y_scale=True, kde=False, kde_bw_adjust=1.5, panel_label='a')

# --- Predictions for Specific Drugs ---
drugs = ['dactinomycin', 'Cadazolid', 'Cefazolin (sodium)', 'Cefodizime', 'Cefotetan', 
         'cefotiam-cilexetil', 'Cefozopran', 'Ceftiofur', 'Levofloxacin', 'Linaclotide', 'Thiostrepton']

print("\n--- Predictions for Specific Drugs ---")
for drug in drugs:
    print(f"\n{drug}:")

    # Ensemble (MoLFormer + D-MPNN Average)
    ens_res = ensemble_df[(ensemble_df['vendor_name'].str.lower() == drug.lower()) | (ensemble_df['pert_iname'].str.lower() == drug.lower())]
    if not ens_res.empty:
        print(f"  Ensemble:      Average Prediction={ens_res['average_prediction'].values[0]:.3f}, Rank={ens_res['rank'].values[0]+1}")
    else:
        print("  Ensemble:      Not found")
        
    # MoLFormer
    mol_res = molformer_df[(molformer_df['vendor_name'].str.lower() == drug.lower()) | (molformer_df['pert_iname'].str.lower() == drug.lower())]
    if not mol_res.empty:
        print(f"  MoLFormer:     Prediction={mol_res['prediction'].values[0]:.3f}, Rank={mol_res['rank'].values[0]+1}, Uncertainty={mol_res['uncertainty'].values[0]:.3f}")
    else:
        print("  MoLFormer:     Not found")

    # D-MPNN
    che_res = chemprop_df[(chemprop_df['vendor_name'].str.lower() == drug.lower()) | (chemprop_df['pert_iname'].str.lower() == drug.lower())]
    if not che_res.empty:
        rank_info = f"Rank={che_res['rank'].values[0]+1}"
        uncertainty_info = f", Uncertainty={che_res['uncertainty'].values[0]:.3f}" if 'uncertainty' in che_res.columns else ""
        print(f"  D-MPNN:        Prediction={che_res['prediction'].values[0]:.3f}, {rank_info}{uncertainty_info}")
    else:
        print("  D-MPNN:        Not found")

    # Random Forest
    rf_res = rf_df[(rf_df['vendor_name'].str.lower() == drug.lower()) | (rf_df['pert_iname'].str.lower() == drug.lower())]
    if not rf_res.empty:
        rank_info = f"Rank={rf_res['rank'].values[0]+1}"
        uncertainty_info = f", Uncertainty={rf_res['uncertainty'].values[0]:.3f}" if 'uncertainty' in rf_res.columns else ""
        print(f"  Random Forest: Prediction={rf_res['prediction'].values[0]:.3f}, {rank_info}{uncertainty_info}")
    else:
        print("  Random Forest: Not found")
    