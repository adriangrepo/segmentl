import torch.nn as nn

class DummyLoss(nn.Module):
    '''
    Dummy Loss for debugging
    '''
    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, inp, target):
        delta = inp-target
        print(f'DummyLoss pred: {inp.shape}, target: {target.shape}, delta: {delta.mean()}')
        return delta.mean()