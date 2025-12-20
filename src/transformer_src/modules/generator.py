import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)
    
class Generator_aa_types_1(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_aa_types_1, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)
    

class Generator_aa_types_2(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_aa_types_2, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)


class Generator_aa_types_3(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_aa_types_3, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)

class Generator_aa_idxs_1(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_aa_idxs_1, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)
    

class Generator_aa_idxs_2(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_aa_idxs_2, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)
    

class Generator_aa_idxs_3(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_aa_idxs_3, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)

class Generator_inter_types_1(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_inter_types_1, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)
    

class Generator_inter_types_2(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_inter_types_2, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)
    

class Generator_inter_types_3(nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator_inter_types_3, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return self.ff(x)