import pandas as pd
from rdkit import Chem

number_of_molecules = 11

# Read predictions file
df = pd.read_csv('results/molformer_predictions.csv')

# Read resistant dataset
resistant_df = pd.read_csv('processed_datasets/10uM_FP_clustered_resistant_pneumococcus_augmented_dataset.csv', usecols=['InChIKey'])

# Generate InChIKeys for resistant set
resistant_inchikeys = set(resistant_df['InChIKey'].dropna())

# Filter out molecules present in resistant dataset by InChIKey
print("Number of molecules in predictions file: ", len(df))
filtered_df = df[~df['InChIKey'].isin(resistant_inchikeys)]
print("Number of molecules in predictions file after filtering: ", len(filtered_df))

# Add Linaclotide if not present
if not (filtered_df['vendor_name'].str.lower() == 'linaclotide').any():
    linaclotide_rows = df[df['vendor_name'].str.lower() == 'linaclotide']
    if not linaclotide_rows.empty:
        linaclotide_rows = linaclotide_rows.assign(rank=None)
        filtered_df = pd.concat([filtered_df, linaclotide_rows], ignore_index=True)

ordered_df = filtered_df.sort_values(by='prediction', ascending=False)
ordered_df = ordered_df.reset_index(drop=True)
ordered_df['rank'] = ordered_df.index

top_n = ordered_df.head(number_of_molecules)

print(f"\nTop {number_of_molecules} predictions (excluding resistant set by InChIKey):")
print(top_n[['vendor_name', 'pert_iname', 'prediction', 'uncertainty']])

# List of drugs to check
drugs = ['Actinomycin D', ### I think this is the same as Actinomycin
         'Cadazolid', 
         'Cefazolin (sodium)', 
         'Cefodizime', 
         'Cefotetan', 
         'Cefotiam-Hexetil', ### in the paper it is written as Cefotiam-Hexetil~, I can't find it in the dataset
         'Cefozopran', 
         'Ceftiofur', 
         'Levofloxacin', 
         'Linaclotide', 
         'Thiostrepton']

print("\nPredictions for specific drugs:")
for drug in drugs:
    drug_data = ordered_df[ordered_df['vendor_name'].str.lower() == drug.lower()]
    drug_data2 = ordered_df[ordered_df['pert_iname'].str.lower() == drug.lower()]
    if not drug_data.empty:
        print(f"{drug}: prediction={drug_data['prediction'].values[0]:.3f}, rank={drug_data['rank'].values[0]}, uncertainty={drug_data['uncertainty'].values[0]:.3f}")
    elif not drug_data2.empty:
        print(f"{drug}: prediction={drug_data2['prediction'].values[0]:.3f}, rank={drug_data2['rank'].values[0]}, uncertainty={drug_data2['uncertainty'].values[0]:.3f}")
    else:
        print(f"{drug}: Not found in dataset")
