# As simple as possible training loop.

import torch
import numpy as np
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.shallow_autoencoder import ConvAutoencoder
