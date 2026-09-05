import torch
import torch.nn as nn
from torch.autograd import Function


class ChockerFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.alpha
        return grad_input, None


class GradChoker(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        alpha = torch.tensor(self.alpha, requires_grad=False, device=x.device)
        return ChockerFunction.apply(x, alpha)
