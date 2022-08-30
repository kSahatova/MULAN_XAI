from sklearn.tree import DecisionTreeRegressor
from captum._utils.models.model import Model
from typing import Optional
from torch import Tensor
import numpy as np
import torch
import time 


def train_tree(model, dataloader,construct_kwargs, norm_input=False,  **fit_kwargs):
    num_batches = 0
    xs, ys, ws = [], [], []
    for data in dataloader:
        #print(data)
        x, y, w = data
        #w = None

        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
        if w is not None:
            ws.append(w.cpu().numpy())
        num_batches += 1

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if len(ws) > 0:
        w = np.concatenate(ws, axis=0)
    else:
        w = None
    if norm_input:
        mean, std = x.mean(0), x.std(0)
        x -= mean
        x /= std

    sklearn_model = DecisionTreeRegressor(**construct_kwargs)
    sklearn_model.fit(x, y)
        
    # Convert weights to pytorch
    classes = (
        torch.IntTensor(sklearn_model.classes_)
        if hasattr(sklearn_model, "classes_")
        else None
    )

    # extract model device
    device = model.device if hasattr(model, "device") else "cpu"

    tree = sklearn_model.tree_ # torch.FloatTensor(sklearn_model.tree_).to(device)  # type: ignore
    model._construct_model_params(
        norm_type=None,
        tree_str=tree,
        classes=classes,
    )

    if norm_input:
        model.norm = NormLayer(mean, std)

    return model 
    
    
class DecisionTree:
    def __init__(self, train_fn, **kwargs):
        self.train_fn = train_fn 
        self.norm: Optional[nn.Module] = None
        self.model: Optional[nn.Linear] = None
        self.construct_kwargs = kwargs
        
    def _construct_model_params(
        self,
        norm_type: Optional[str] = None,
        affine_norm: bool = False,
        tree_str: Optional[Tensor] = None,
        classes: Optional[Tensor] = None,
    ):
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(in_features, eps=1e-8, affine=affine_norm)
        elif norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                in_features, eps=1e-8, elementwise_affine=affine_norm
            )
        else:
            self.norm = None

        self.model = DecisionTreeRegressor()

        if tree_str is not None:
            self.model.tree_ = tree_str

        if classes is not None:
            self.model.classes = classes
                
    def fit(self, train_data, **kwargs):
        return self.train_fn(
            self,
            dataloader=train_data,
            construct_kwargs=self.construct_kwargs,
            **kwargs
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert self.linear is not None
        if self.norm is not None:
            x = self.norm(x)
        return self.linear(x)
    
    def representation(self):
        r"""
        Returns a tensor which describes the hyper-plane input space. This does
        not include the bias. For bias/intercept, please use `self.bias`
        """
        assert self.model is not None
        return torch.from_numpy(self.model.feature_importances_) 