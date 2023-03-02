# ------------------------------------------------------------------------------
# @file:    mlp.py
# @brief:   This file contains the implementation of a Multi-Layer Perceptron 
#           (MLP) network. 
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 3rd, 2022
# ------------------------------------------------------------------------------

# general includes
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

seed = 1
np.random.seed(seed)


class MLP(nn.Module):
    def __init__(
        self, config: dict, device: str = "cuda:0", softmax: bool = False
    ) -> None:
        """ Implments a simple MLP with ReLU activations. 
        Inputs:
        -------
        config[dict]: network configuration parameters.
        device[str]: device used by the module. 
        """
        self._name = self.__class__.__name__
        super(MLP, self).__init__()
        
        self._config = config
        logger.debug("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)))
        
        self.device = device
        logger.debug(f"{self.name} uses torch.device({self.device})")
        
        self.dropout = self.config.dropout
        self.layer_norm = 'layer_norm' in self.config and self.config.layer_norm
        
        # ----------------------------------------------------------------------
        # Network architecture 
        feats = [config.in_size, *config.hidden_size, config.out_size]
        mlp = []
        mlp_norm = []
        for i in range(len(feats)-1):
            mlp.append(
                nn.Linear(in_features=feats[i], out_features=feats[i+1])
            )
            mlp_norm.append(
                nn.LayerNorm(feats[i+1])
            )
            
        if softmax:
            mlp.append(nn.Softmax(dim=-1))
            
        self.net = nn.ModuleList(mlp)
        self.net_norm = nn.ModuleList(mlp_norm) if self.layer_norm else None
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self)-> dict:
        return self._config
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Forward propagation of x.
        Inputs:
        -------
        x[torch.tensor(batch_size, input_size)]: input tensor
            
        Outputs:
        -------
        x[torch.tensor(batch_size, output_size)]: output tensor
        """ 
        for i in range(len(self.net)-1):
            x = self.net[i](x)
            if self.net_norm:
                x = self.net_norm[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.net[-1](x)
        return x

class MultiMLP(nn.Module):
    def __init__(
        self, config: dict, device: str = "cuda:0", softmax: bool = False
    ) -> None:
        self._name = self.__class__.__name__
        super(MultiMLP, self).__init__()
        
        self._config = config
        logger.debug("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)))
        
        self.device = device
        logger.debug(f"{self.name} uses torch.device({self.device})")

        self.n_files = self._config.n_files
        self.mlps = nn.ModuleList([MLP(config, device=device) for _ in range(self.n_files)])
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self)-> dict:
        return self._config
    
    def forward(self, x: torch.tensor, x_idx: torch.tensor) -> torch.tensor:
        ret = torch.stack([self.mlps[idx](x_) for idx, x_ in zip(x_idx, x)])
        return ret


