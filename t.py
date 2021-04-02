import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets

device = 'cuda:0'


class CPC(nn.Module):
    def __init__(self, batch_size, timestep):

        super().__init__()
        self.batch_size = batch_size

        self.timestep = timestep
        self.W = nn.ModuleList([nn.Linear(77, 55) for i in range(timestep)])

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.device = 'cpu'

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, x):
        pred = torch.empty((self.batch_size, self.timestep, 55)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.W[i]
            pred[:, i] = linear(x[:, i])  # Wk*c_t e.g. size 8*512

        return pred


x = torch.randn(16, 12, 77)
A = torch.randn(77, 55)
y = torch.matmul(x, A)

model = CPC(x.size(0), x.size(1))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

x = x.to(device)
y = y.to(device)
for i in range(10):
    optimizer.zero_grad()
    pred = model(x)
    #     print(f'model.pred={model.pred}')
    loss = loss_fn(pred, y)
    print(loss)
    loss.backward()
    optimizer.step()
