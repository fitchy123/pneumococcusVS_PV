import torch 
import torch.nn as nn 
from typing import Any, Mapping
import lightning as L

from torcheval.metrics.functional import binary_auprc
from torchmetrics.functional.classification import binary_matthews_corrcoef
from transformers import AutoModel, AutoTokenizer
# from losses import WeightedFocalLoss

class FFN(nn.Module):
    def __init__(
            self,
            input_dim = 768,
            hidden_dim = 1024,
            output_dim = 2,
            n_layers:int = 1,
            dropout:float = 0.3,
            activation:nn.modules.activation = nn.GELU(),
        ):
        """
        Feed-forward layers to be added on top of MolFormer's transformer blocks
        hidden_dim: hidden dimension of the FFN
        output_dim: output dimension of the FFN
        n_layers: number of FFN layers
        dropout: dropout rate of the FFN layers
        """
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

        if n_layers == 1:
            self.linears: Any = [nn.Linear(input_dim, output_dim)]

        else:
            self.linears = [nn.Linear(input_dim, hidden_dim)] + [
                nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
            ] + [nn.Linear(hidden_dim, output_dim)]

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers == 1:
            return self.dropout_p[0](self.activation(self.linears[0](x)))
        else:
            # x = self.dropout_p[0](x)    # no activation at input
            for i in range(self.n_layers - 1):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))
            return self.linears[-1](x)
        
class FFNMolFormer(L.LightningModule):
    def __init__(
                self,
                is_frozen=True,
                hidden_dim = 1024,
                output_dim = 2,
                n_layers:int = 1,
                dropout:float = 0.3,
                # activation:nn.modules.activation = nn.GELU(),
                # loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
        ):
        """
        MolFormer's embedding layer + transformer blocks and additional feed forward layers
        is_frozen: whether to freeze the weight of the transformer blocks
        hidden_dim: hidden dimension of the FFN
        output_dim: output dimension of the FFN
        n_layers: number of FFN layers
        dropout: dropout rate of the FFN layers
        """
        super(FFNMolFormer, self).__init__()
        self.base = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.ffn = FFN(
                input_dim=768,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                dropout=dropout,
                activation=nn.GELU()
            )
        self.loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
        self.save_hyperparameters()
        self.y_true = []
        self.eval_y_scores = []
        self.eval_loss = []

        # Freeze MolFormer params
        if is_frozen:
            for param in self.base.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask).pooler_output
        outputs = self.ffn(outputs)
        return outputs
    
    def training_step(self, batch) -> torch.Tensor:
        X, labels = batch
        pred = self.forward(**X)
        y = labels.view(pred.shape).to(torch.float64)

        loss = self.loss_fn(pred.double(), y)
        auprc = binary_auprc(pred[:,1], y[:,1])
        self.log("train_auprc", auprc)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001, weight_decay=0.1)
        return optim
    
    def validation_step(self, val_batch) -> torch.Tensor:
        X, labels = val_batch
        pred = self.forward(**X)
        y = labels.view(pred.shape).to(torch.float64)

        self.y_true.append(labels.view(pred.shape))
        self.eval_y_scores.append(pred)

        #Whether y is non-null or not.
        is_valid = ~torch.isnan(torch.sum(y, 1))
        #Loss matrix
        loss_mat = self.loss_fn(pred.to(torch.float32), y)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        self.eval_loss.append(loss)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        y_true = torch.cat(self.y_true, dim = 0)
        y_scores = torch.cat(self.eval_y_scores, dim = 0)
        auprc = binary_auprc(y_scores[:,1], y_true[:,1])
        self.log("val_auprc", auprc, sync_dist=True)
        mcc = binary_matthews_corrcoef(y_scores[:,1], y_true[:,1], threshold=0.5)
        self.log("val_mcc50", mcc, sync_dist=True)
        
        self.eval_loss.clear()
        self.y_true.clear()
        self.eval_y_scores.clear()