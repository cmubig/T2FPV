import torch.nn as nn

class ETHUCYFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.layer_norm = 'layer_norm' in args and args.layer_norm
        if self.layer_norm:
            self.embed = nn.Sequential(nn.Linear(args.input_dim, self.embbed_size), 
                                       nn.LayerNorm(self.embbed_size),
                                            nn.ReLU()) 
        else:
            self.embed = nn.Sequential(nn.Linear(args.input_dim, self.embbed_size), 
                                            nn.ReLU()) 

    def forward(self, inputs):
        box_input = inputs

        embedded_box_input= self.embed(box_input)

        return embedded_box_input

def build_feature_extractor(args):
    return ETHUCYFeatureExtractor(args)