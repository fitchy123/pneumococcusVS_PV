from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, auc, matthews_corrcoef, precision_score, recall_score
from imblearn.ensemble import BalancedRandomForestClassifier
import argparse
import os

# Update argument parsing
parser = argparse.ArgumentParser(description='Run Random Forest model with different options')
parser.add_argument('--load_model', default=True, type=bool, help='Load a previously trained model')
parser.add_argument('--tuning', action='store_true', help='Perform hyperparameter tuning')
parser.add_argument('--balanced', action='store_true', help='Use balanced random forest')
parser.add_argument('--model_path', type=str, default='model_checkpoints/tuned_rfc_morgan_seed3.joblib',
                    help='Path to load/save the model')
parser.add_argument('--data_path', type=str, 
                    default='processed_datasets/repurposing_samples.csv',
                    help='Path to the input dataset')
parser.add_argument('--output_path', type=str, default='results/rf_predictions.csv', help='Path to save predictions')
parser.add_argument('--train', action='store_true', help='Train')
args = parser.parse_args()

# read in data
df_all = pd.read_csv(args.data_path)

# Filter for valid SMILES
initial_rows = len(df_all)
valid_smiles_mask = df_all['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
df = df_all[valid_smiles_mask].copy()
filtered_rows = len(df)
if filtered_rows < initial_rows:
    print(f"Removed {initial_rows - filtered_rows} rows with invalid SMILES strings.")

if args.train:
    train_idx = df[df.cluster_id!=-1].index.values 
    test_idx = df[df.cluster_id==-1].index.values
    # generate 2048-bit morgan fingerprint of radius 3
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, 2048, useChirality=True) for smiles in df.loc[train_idx]['smiles'].values]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, 2048, useChirality=True) for smiles in df.loc[test_idx]['smiles'].values]
else:
    test_idx = df.index.values
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, 2048, useChirality=True) for smiles in df.loc[test_idx]['smiles'].values]




if args.load_model:
    rfc_cv = joblib.load(args.model_path)
else:
    if args.tuning:
        print("Tuning hyperparameters")
        # hyperparameter tuning
        params = {'max_depth':[None, 16, 256],
                'min_samples_leaf':[1, 2, 4],
                'criterion':['gini','entropy'],
                'class_weight':['balanced',None]}

        rfc = RandomForestClassifier(n_estimators=200, random_state=3) #32
        rfc_cv = GridSearchCV(rfc, params, cv =5, n_jobs=5, scoring='average_precision', verbose=2)
        rfc_cv.fit(train_fps, df.loc[train_idx]['final_activity_label'])
        joblib.dump(rfc_cv.best_estimator_, args.model_path)

        rfc_cv_resultsDF = pd.DataFrame(rfc_cv.cv_results_).sort_values('mean_test_score', ascending=False)
        rfc_cv_resultsDF.to_csv(r'processed_datasets/rfc_morgan_5StratifiedFold_CV_results_repro.csv', index=False)
        print(rfc_cv.best_estimator_)
    elif args.balanced:
        print("Training singular balanced model")
        rfc_cv = BalancedRandomForestClassifier(n_estimators=200, random_state=3).fit(train_fps, df.loc[train_idx]['final_activity_label'])
    else:
        print("Training singular model")
        rfc_cv = RandomForestClassifier(n_estimators=200, random_state=3).fit(train_fps, df.loc[train_idx]['final_activity_label'])

pred = [x[1] for x in rfc_cv.predict_proba(test_fps)]

if args.output_path:
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_all['prediction'] = -1.0
    df_all['uncertainty'] = 0.0
    df_all.loc[test_idx, 'prediction'] = pred
    df_all.to_csv(args.output_path, index=False)

#p_precision, p_recall, thresholds = precision_recall_curve(df['final_activity_label'].loc[test_idx], pred)
#print('\tMCC @0.5: ', matthews_corrcoef(df['final_activity_label'].loc[test_idx], [x >0.5 for x in pred]))
#print('\tprecision: ', precision_score(df['final_activity_label'].loc[test_idx], [x >0.5 for x in pred]))
#print('\trecall: ', recall_score(df['final_activity_label'].loc[test_idx], [x >0.5 for x in pred]))
#print('\tAUPRC: ', auc(p_recall, p_precision))
#print('\tROC-AUC score: ', roc_auc_score(df['final_activity_label'].loc[test_idx], pred))