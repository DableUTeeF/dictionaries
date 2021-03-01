import torch

if __name__ == '__main__':
    a1 = torch.rand([32, 8, 28, 28])
    a2 = torch.rand([32, 8, 28, 28])

    tmp1 = torch.sum([a1, a2], dim=[1, 2, 3])

