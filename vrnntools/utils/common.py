# ------------------------------------------------------------------------------
# @file:    common.py
# @brief:   This file contains the implementation common utils needed by sprnn.
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 3rd, 2022
# ------------------------------------------------------------------------------

import logging
import torch
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

# definitions below
DIMS = [2, 3]
COORDS = ["rel", "abs"]
TRAJ_ENCODING_TYPE = ["mlp", "tcn"]
ADJ_TYPE = ["fc", "fully-connected", "distance-similarity", "knn", "gaze"]
FORMAT = '[%(asctime)s: %(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'

# classes below:

class Config:
    """ A class for holding configuration parameters. """
    def __init__(self, config):
        self.BASE_CONFIG = dotdict(config)
        
        self.GOAL = None
        if self.BASE_CONFIG.goal:
            self.GOAL = dotdict(self.BASE_CONFIG.goal)
            
        self.TRAJECTORY = None
        if self.BASE_CONFIG.trajectory:
            self.TRAJECTORY = dotdict(self.BASE_CONFIG.trajectory)

        self.DATASET = None
        if self.BASE_CONFIG.dataset:
            self.DATASET = dotdict(self.BASE_CONFIG.dataset)

        self.TRAIN = None
        if self.BASE_CONFIG.training_details:
            self.TRAIN = dotdict(self.BASE_CONFIG.training_details)

        self.MODEL = None
        if self.BASE_CONFIG.model_design:
            self.MODEL = dotdict(self.BASE_CONFIG.model_design)

        self.VISUALIZATION = None
        if self.BASE_CONFIG.visualization:
            self.VISUALIZATION = dotdict(self.BASE_CONFIG.visualization)
            
        self.MAP = None
        if self.BASE_CONFIG.map:
            self.MAP = dotdict(self.BASE_CONFIG.map)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# methods below:

def convert_rel_to_abs(traj_rel, start_pos, permute: bool = False):
    """ Converts a trajectory expressed in relative displacement to an absolute
    values given a start position. 
    
    Inputs:
    -------
    traj_rel[torch.tensor(batch, seq_len, dim)]: trajectory of displacements
    start_pos[torch.tensor(batch, dim)]: initial absolute position
   
    Outputs:
    --------
    traj_abs[torch.tensor(seq_len, batch, 2)]: trajectory of absolute coords
    """
    if permute:
        # (seq_len, batch, 2) -> (batch, seq_len, 2)
        traj_rel = traj_rel.permute(1, 0, 2)        
        
    displacement = torch.cumsum(traj_rel, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    
    if permute:
        return abs_traj.permute(1, 0, 2)
    return abs_traj

# From https://stackoverflow.com/a/62508086
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
