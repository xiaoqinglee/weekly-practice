import torch

def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(rms_norm(x, torch.ones(4)))
