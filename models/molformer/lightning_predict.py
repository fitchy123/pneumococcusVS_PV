import pandas as pd
import torch
import torch.nn as nn
import glob

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

import os, argparse
from datetime import datetime
import numpy as np
from typing import Any
from tqdm import tqdm
from functools import partial
from tqdm import tqdm

from dataset import MoleculeDataset, smilesCollate
from lightning_model import FFN, FFNMolFormer
from lightning.pytorch import Trainer, seed_everything
from deepchem.feat import RDKitDescriptors
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, auc, matthews_corrcoef, precision_score, recall_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'


MolFormerXL = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

def predictSmilesCollate(smiles, tokenizer, max_len=800, get_global_features=False):
    smiles_inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', max_length=max_len)
    if get_global_features:
        featurizer = RDKitDescriptors(use_bcut2d=False, is_normalized=True)
        global_features = torch.tensor(featurizer.featurize(smiles)).to(torch.float32)
        global_features = torch.nan_to_num(global_features)
        return smiles_inputs, global_features
    
    return smiles_inputs

if __name__ == '__main__':
    print("start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'processed_datasets/repurposing_samples.csv', type=str, help='dataset file path')
    parser.add_argument('--batch_size', default=256,  type=int, help='train/test batch size')
    parser.add_argument('--hidden_dim', default=1536,  type=int, help='FFN hidden dim')
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--dev_run', action='store_true', help='run a debug test')
    parser.add_argument('--checkpoint_path', default='model_checkpoints/TunedCEMolFormerSeed1', type=str)
    parser.add_argument('--num_workers', default=4,  type=int, help='number of dataloader workers')
    parser.add_argument('--agg_fn', type=str, default='median', choices=['mean', 'median'], help='ensemble aggregation function')
    parser.add_argument('--output_path', default=r'results/molformer_predictions.csv', type=str)
    parser.add_argument('--data_limit', default=None, type=int)
    # additional argument if you only want to run models on test data (cluster -1)
    parser.add_argument('--test_only', action='store_true', help='Run only on test data')
    # additional cluster id colname argument for convenience
    parser.add_argument('--cluster_id_colname', default='cluster_id',  type=str, help='name of the column containing cluster ID')

    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")

    df = pd.read_csv(args.data_path)
    if args.data_limit is not None:
        df = df.iloc[:args.data_limit]
    
    # added code for test only flag
    if args.test_only:
        test_fold = [-1]
        test_idx = df[df[args.cluster_id_colname].isin(test_fold)].index.values
        df = df[df.index.isin(test_idx)]

    smiles_collater = partial(predictSmilesCollate, tokenizer=MolFormerXL_tokenizer, max_len=800)
    seed_everything(args.random_seed)
    # torch.autograd.detect_anomaly(True)

    test_dataset = MoleculeDataset(df['smiles'].values, label=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=smiles_collater, num_workers=args.num_workers)

    model_files = glob.glob(rf'{args.checkpoint_path}*')    # load ensembles of molformer models
    print("inference start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " with ", len(model_files), " models")
    with torch.no_grad():
        ensemble_y_scores = []
        for model_path in tqdm(model_files):
            model = FFNMolFormer.load_from_checkpoint(model_path).to(device='cuda')
            model.eval()
            y_scores = []
            for batch in tqdm(test_loader, leave=False):
                batch = batch.to(device='cuda')
                output = model(**batch)
                # print(output.shape)
                y_scores.append(output.cpu())
            y_scores = torch.cat(y_scores, dim = 0)
            ensemble_y_scores.append(y_scores[:,1])
        
        ensemble_y_scores = torch.vstack(ensemble_y_scores)
        pred_prob = 1 / (1 + torch.exp(-ensemble_y_scores)) # sigmoid function
        if args.agg_fn == 'median':
            # aggregate with scores median and use interquartile range as measure of uncertainty
            agg_y_scores = torch.median(pred_prob, dim=0).values
            uncertainty = torch.quantile(pred_prob, torch.tensor([0.25, 0.75]), dim=0)
            uncertainty = uncertainty[1,:] - uncertainty[0,:]
        elif args.agg_fn == 'mean':
            # aggregate with scores mean and use standard deviation as measure of uncertainty
            agg_y_scores = torch.mean(pred_prob, dim=0)
            # pred_prob = 1 / (1 + torch.exp(-agg_y_scores))
            uncertainty = torch.std(pred_prob, dim=0)
        else:
            print('please input a valid aggregation')
    
    print("inference end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    df['prediction'] = agg_y_scores.detach().numpy()
    df['uncertainty'] = uncertainty.detach().numpy()

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        df.to_csv(args.output_path, index=False)

    """print('\tMCC @top1%: ', matthews_corrcoef(df['final_activity_label'], [x>0.5 for x in df['prediction']]))
    print('\tprecision: ', precision_score(df['final_activity_label'], [x>0.5 for x in df['prediction']]))
    print('\trecall: ', recall_score(df['final_activity_label'], [x>0.5 for x in df['prediction']]))
    p_precision, p_recall, thresholds = precision_recall_curve(df['final_activity_label'], df['prediction'])
    print('\tAUPRC: ', auc(p_recall, p_precision))

    if args.output_path is not None:
        df[['Smiles', 'cluster_id', 'prediction', 'uncertainty']].to_csv(os.path.join('processed_datasets', args.output_path, '.csv'), index=False)
    
    """
    print("finish time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

