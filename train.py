# As-simple-as-possible training loop.

import torch
import numpy as np
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.shallow_autoencoder import ConvAutoencoder


# load model definition
model = ConvAutoencoder()
model = model.double()  # tackles a type error

# define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# prepare pytorch dataloader
X = np.random.random((300, 1, 100))  # toy data
Y = X
dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Training loop
for epoch in range(200):
    for x, y in dataloader:
        
        optimizer.zero_grad()
        
        # forward and backward pass
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        print(loss.item())  # loss should be decreasing
