import torch 
import torch.optim as optim
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from collections import deque
from torchvision import datasets, transforms