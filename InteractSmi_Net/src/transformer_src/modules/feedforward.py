import torch
import torch.nn as nn
from torch.nn import functional as F




class FFN (nn.Module):
    '''前馈神经网络'''
    def __init__(self, model_depth, ff_depth, dropout):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(model_depth, ff_depth)
        self.w2 = nn.Linear(ff_depth, model_depth)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))