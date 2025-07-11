from deepchem.data import Dataset
from deepchem.models import Model
from typing import List, Callable, Tuple

import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, confusion_matrix, auc, roc_auc_score

def _get_model_prediction(model, dataset: Dataset, agg_fn:str):
    """
    wrapper to get prediction from sklearn models, deepchem models, and model ensembles (as list)
    """
    def predict(model, dataset):
        if isinstance(model, Model):
            pred = model.predict(dataset)
        else:
            pred = model.predict_proba(dataset.X)
        pred = [x[1] for x in pred]
        return pred

    if isinstance(model, List):
        predictions = np.empty((len(model), dataset.y.shape[0]))
        if isinstance(model[0], Tuple):
            model = [x[0] for x in model]
        for ensemble_id, submodel in enumerate(model):
            internal_pred = predict(submodel, dataset)
            predictions[ensemble_id] = internal_pred
        
        if agg_fn == 'mean':
            pred = np.mean(predictions, axis=0)
            uncertainty = np.quantile(predictions, 0.75, axis = 0) - np.quantile(predictions, 0.25, axis = 0)
        elif agg_fn == 'median':
            pred = np.median(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
        else:
            print('Not Implemented')
            return None
        return pred, uncertainty
    else:
        pred = predict(model, dataset)
        return pred, np.full_like(pred, np.nan)