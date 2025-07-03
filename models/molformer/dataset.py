from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from deepchem.feat import RDKitDescriptors
import torch
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, src, label=None, label_type='classification'):
        self.src = src
        # self.label = label.reshape(-1, 1)
        if label is not None and label_type=='classification':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.label = encoder.fit_transform(label.reshape(-1, 1))
        else:
            self.label=label

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        if self.label is not None:
            label = self.label[idx]
            return src, label
        else:
            return src
        
def smilesCollate(samples, tokenizer, max_len=800, get_global_features=False):
    smiles = [x[0] for x in samples]
    labels = [x[1] for x in samples]

    smiles_inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', max_length=max_len)
    labels = torch.tensor(np.array(labels)).type(torch.float32)

    # in case need to add 200 molecular-level rdkit descriptors
    if get_global_features:
        featurizer = RDKitDescriptors(use_bcut2d=False, is_normalized=True)
        global_features = torch.tensor(featurizer.featurize(smiles)).to(torch.float32)
        global_features = torch.nan_to_num(global_features)
        return smiles_inputs, global_features, labels
    
    return smiles_inputs, labels
