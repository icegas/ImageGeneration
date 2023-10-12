import torch
from torch.nn.functional import mse_loss as torch_mse

__all__=['mse_loss']

def mse_loss():
    return MSELoss

class MSELoss(torch.nn.MSELoss):

    def __init__(self, params):
        super().__init__(params)
        self.reduction = params.reduction
    
    @property
    def names(self):
        return ['main_loss']
    
    def forward(self, inputs, targets):
        loss = torch_mse(inputs, targets, reduction=self.reduction)

        return {'main_loss' : loss}