import pandas as pd
from rdkit import Chem
import argparse
import pickle
import glob
import os

from deepchem.feat import DMPNNFeaturizer
from deepchem.data import NumpyDataset
from chemprop import MyDMPNNModel
import torch

from predict_utils import _get_model_prediction
from losses import dcFocalLoss

from datetime import datetime
import numpy as np      
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, precision_score, recall_score


if __name__ == '__main__':
    print("start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'processed_datasets/repurposing_samples.csv', type=str, help='dataset file path')
    parser.add_argument('--cluster_id_colname', default='cluster_id',  type=str, help='name of the column containing cluster ID')
    parser.add_argument('--model_dirs', default='model_checkpoints/_',  type=str, help='filename prefix of the saved model')
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--algo', default='chemprop', nargs='?',  type=str, choices=['chemprop', 'dmpnn'], help='choice of algorithms')
    parser.add_argument('--loss_fn', default='cross_entropy', nargs='?',  type=str, choices=['cross_entropy', 'focal'], help='choice of algorithms')
    parser.add_argument('--focal_alpha', default=0.25,  type=int, help='focal loss alpha, only applicable if loss_fn == "focal"')
    parser.add_argument('--focal_gamma', default=2,  type=int, help='focal loss gamma, only applicable if loss_fn == "focal"')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--data_limit', default=None, type=int)
    parser.add_argument('--output_path', default=r'results/chemprop_predictions.csv', type=str)
    parser.add_argument('--featurized_data_path', default=r'processed_datasets/featurized_data_chemprop.pkl', type=str, help='path to featurized dataset for speed up')
    parser.add_argument('--save_processed_data', action='store_true', help='save processed data')
    parser.add_argument('--test_only', action='store_true', help='Run only on test data')
    parser.add_argument('--train', default=False, type=bool, help='Train')

    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    df = pd.read_csv(args.data_path)
    if args.data_limit is not None:
        df = df.iloc[:args.data_limit]
    
    # added code for test only flag
    if args.test_only:
        test_fold = [-1]
        test_idx = df[df[args.cluster_id_colname].isin(test_fold)].index.values
        predict_df = df[df.index.isin(test_idx)]
    else:
        predict_df = df
    
    if args.algo == 'dmpnn':
        mpnn_featurizer = DMPNNFeaturizer()
        global_feat_size = 0
    elif args.algo == 'chemprop':
        mpnn_featurizer = DMPNNFeaturizer(features_generators=['rdkit_desc_normalized']) # with molecular-level features
        global_feat_size = 200

    valid_smiles_mask = predict_df['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    predict_df_valid = predict_df[valid_smiles_mask]

    if args.featurized_data_path and not args.save_processed_data:
        print(f"Loading precomputed dataset from {args.featurized_data_path}")
        with open(args.featurized_data_path, "rb") as f:
            mpnn_dataset = pickle.load(f)
        
        all_cached_smiles_set = set(mpnn_dataset.ids)
        in_cache_mask = predict_df_valid['smiles'].isin(all_cached_smiles_set)
        predict_df_valid_in_cache = predict_df_valid[in_cache_mask]

        if len(predict_df_valid_in_cache) < len(predict_df_valid):
            print(f"Warning: {len(predict_df_valid) - len(predict_df_valid_in_cache)} valid SMILES from input were not found in the featurized cache and will not be predicted.")
        
        smiles_to_predict = predict_df_valid_in_cache['smiles'].tolist()
        
        # Efficiently get indices for selection
        all_cached_smiles_map = {smile: i for i, smile in enumerate(mpnn_dataset.ids)}
        indices_to_select = [all_cached_smiles_map[s] for s in smiles_to_predict]

        predict_dataset = mpnn_dataset.select(indices_to_select)

    else:
        print("Featurizing data")

        initial_rows = len(predict_df)
        # Filter out invalid SMILES strings that RDKit cannot process.
        filtered_rows = len(predict_df_valid)
        if filtered_rows < initial_rows:
            print(f"Removed {initial_rows - filtered_rows} rows with invalid SMILES strings.")

        mols = [Chem.MolFromSmiles(smiles) for smiles in predict_df_valid["smiles"]]
        features = mpnn_featurizer.featurize(mols)
        if args.train:
            predict_dataset = NumpyDataset(
                X=features, y=predict_df_valid["final_activity_label"],
                ids=predict_df_valid["smiles"]
            )
        else:
            predict_dataset = NumpyDataset(
                X=features, y=None,
                ids=predict_df_valid["smiles"]
            )

        if args.save_processed_data and not args.test_only and not args.data_limit:
            with open(f'processed_datasets/featurized_data_{args.algo}.pkl', "wb") as f:
                pickle.dump(predict_dataset, f)
            print(f"Saved featurized data to processed_datasets/featurized_data_{args.algo}.pkl")
    
    if args.loss_fn == 'focal':
        loss_fn = dcFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        loss_fn=None

    model_files = glob.glob(f'{args.model_dirs}*')
    model_ensembles = []
    print("inference start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f" {len(model_files)} models")
    for model_dir in model_files:
        model = MyDMPNNModel(mode='classification', 
                                # learning_rate=5e-5,
                                n_classes=2,
                                batch_size=256, 
                                # enc_dropout_p=0.4, 
                                ffn_dropout_p=0.3, 
                                ffn_hidden=1600,
                                depth=6,
                                enc_activation='gelu',
                                ffn_activation='gelu',
                                bias=True,
                                aggregation='sum',
                                global_features_size=global_feat_size,
                                # model_dir='model_garden/{args.algo}_{args.loss_fn}_s{args.random_seed}'
                                model_dir=model_dir
                            )
        model.restore()
        model_ensembles.append(model)
    
    
    # print('#### Model Architecture ####')
    # print(model)
    # print("loss fn: ", loss_fn or 'cross_entropy')
    # print('############################')
    
    cp_pred, cp_uncertainty = _get_model_prediction(model_ensembles, predict_dataset, agg_fn='median')
    print("inference end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(args.model_dirs)
    #print('\tMCC @0.5: ', matthews_corrcoef(predict_df['final_activity_label'], [x>0.5 for x in cp_pred]))
    #print('\tprecision: ', precision_score(predict_df['final_activity_label'], [x>0.5 for x in cp_pred]))
    #print('\trecall: ', recall_score(predict_df['final_activity_label'], [x>0.5 for x in cp_pred]))
    #p_precision, p_recall, thresholds = precision_recall_curve(predict_df['final_activity_label'], [x for x in cp_pred])
    #print('\tAUPRC: ', auc(p_recall, p_precision))

    
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        predict_df['prediction'] = -1.0
        predict_df['uncertainty'] = -1.0

        if args.featurized_data_path and not args.save_processed_data:
            # Predictions are only available for valid smiles that were in the cache
            predict_df.loc[predict_df_valid_in_cache.index, 'prediction'] = cp_pred
            predict_df.loc[predict_df_valid_in_cache.index, 'uncertainty'] = cp_uncertainty
        else:
            # Predictions are available for all valid smiles
            predict_df.loc[predict_df_valid.index, 'prediction'] = cp_pred
            predict_df.loc[predict_df_valid.index, 'uncertainty'] = cp_uncertainty
        
        predict_df.to_csv(args.output_path, index=False)
    
    print("end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # best_threshold, best_mcc, dataset_prediction = _get_best_thresholds(
    #                                             model,
    #                                             dataset=mpnn_dataset,
    #                                             test_indices=[train_idx],
    #                                         )

    # fold_test_mcc, test_bestMCC, conf_matrix, rocauc, auprc = 
    # check_result(model_ensembles, 
    #             test_dataset, 
    #             agg_fn=np.median, 
    #             thresh=0.5)
