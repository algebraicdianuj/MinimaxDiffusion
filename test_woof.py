import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='synthetic image folder')
args = parser.parse_args()

def test(model, data_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))
        z = self.fc4(y)
        return z
    
    def feature(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))
        return y

batch_size = 256
num_classes = 10 
channel = 3
ipc = int(args.dir.split('_')[-1].split('_')[-1])
im_size = (256, 256)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Resize((im_size[0], im_size[1]))])
train_dataset = torchvision.datasets.ImageFolder(root='image_woof_dataset', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Resize((im_size[0], im_size[1]))])

# Replace CIFAR10 dataset with the new dataset
syn_dataset = torchvision.datasets.ImageFolder(root=args.dir, transform=transform)
syn_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

testing_net = MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)
optim_testing_net = torch.optim.Adam(testing_net.parameters(), lr=1e-3)

for _ in range(30):
    for batch in syn_loader:
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        optim_testing_net.zero_grad()
        output = testing_net(imgs)
        loss_test = nn.CrossEntropyLoss()(output, labels)
        loss_test.backward()
        optim_testing_net.step()

test_acc = test(testing_net, train_loader, device)
print("Testing accuracy (%): ", test_acc)

# Save test accuracy to a file
with open(f'testingacc_ipc_{ipc}.txt', 'w') as f:
    f.write(f'{test_acc:.2f} percent')

