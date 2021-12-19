import torch

class BinarizedF(torch.autograd.Function):

    # @staticmethod
    def forward(self, input_):

        # self.save_for_backward(input_)
        a = torch.ones_like(input_)
        b = torch.zeros_like(input_)
        output = torch.where(input_>0.5, a, b)

        return output

    # @staticmethod
    def backward(self, grad_output):
  
        return grad_output

