import pandas as pd

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

# Read resistant dataset and generate InChIKeys for filtering
resistant_df = pd.read_csv('processed_datasets/10uM_FP_clustered_resistant_pneumococcus_augmented_dataset.csv')
resistant_df = resistant_df[resistant_df['source']!='MRSA_Wong2024']
resistant_inchikeys = set(resistant_df['InChIKey'].dropna())

# Process predictions for each model
molformer_df = process_predictions('results/molformer_predictions.csv', resistant_inchikeys, "Molformer")
chemprop_df = process_predictions('results/chemprop_predictions.csv', resistant_inchikeys, "Chemprop")
rf_df = process_predictions('results/rf_predictions.csv', resistant_inchikeys, "Random Forest")

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
    