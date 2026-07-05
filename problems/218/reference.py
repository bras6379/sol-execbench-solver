import torch

def run(A, B):
    C = torch.matmul(A, B.T)
    return C