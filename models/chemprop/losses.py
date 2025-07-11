import torch 
from deepchem.models.losses import Loss
from typing import List 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device=device)
        self.gamma = torch.tensor(gamma).to(device=device)

    def forward(self, inputs, targets):
        targets = targets.type(torch.long)
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')

        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
class dcFocalLoss(Loss):
    def __init__(self, alpha, gamma):
        super(dcFocalLoss).__init__()
        self.criterion = self._create_pytorch_loss()
        self.alpha = alpha 
        self.gamma = gamma

    def _create_pytorch_loss(self):
        def loss(output, labels):
            # output, labels = _make_pytorch_shapes_consistent(output, labels)
            focal_loss = WeightedFocalLoss(alpha=self.alpha, gamma=self.gamma)
            return focal_loss(output, labels)
        return loss
    
    def __call__(self, outputs: List, labels: List, weights: List) -> float:
        if len(outputs) != 1 or len(labels) != 1 or len(weights) != 1:
            raise ValueError(
                "Loss functions expects exactly one each of outputs, labels, and weights"
            )
        losses = self.criterion(outputs[0], labels[0])
        # w = weights[0]
        # if len(w.shape) < len(losses.shape):
        #     if isinstance(w, torch.Tensor):
        #         shape = tuple(w.shape)
        #     else:
        #         shape = w.shape
        #     shape = tuple(-1 if x is None else x for x in shape)
        #     w = w.reshape(shape + (1,) * (len(losses.shape) - len(w.shape)))

        # loss = losses * w
        loss = losses.mean()
        return loss