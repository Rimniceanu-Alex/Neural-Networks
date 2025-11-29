import torch
import torchvision.transforms as transforms # Transformations we can perform on our dataset for augmentation
from torch import optim # For optimizers like SGD, Adam, etc.
from torch import nn # To inherit our neural network
from torch.utils.data import DataLoader # For management of the dataset (batches)
from tqdm import tqdm # For nice progress bar!
from torch.utils.data import Dataset
import pickle
import os
import pandas as pd
import numpy as np

class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str="/kaggle/input/fii-nn-2025-homework-4", train: bool = True):
        self.file = "extended_mnist_test.pkl"
        if train:
            self.file = "extended_mnist_train.pkl"
        self.file = os.path.join(root, self.file)
        with open(self.file, "rb") as fp:
            self.data = pickle.load(fp)
        self.transform=transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i : int):
        first=self.data[i]
        reshape_image=first[0].reshape(28,28)
        image=self.transform(reshape_image)
        return image, self.data[i][1]



class NN(nn.Module):
    def __init__(self, input_size, num_classes):
 
        super(NN, self).__init__()
 
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(1024)
        self.norm2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias) 
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias) 
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias) 
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x

def solve(loader, model):
    model.eval()
    predictions_csv = {
    "ID": [],
    "target": []
    }
    with torch.no_grad():
        count=0
        for x, _ in loader:
            x = x.to(device=device)
            x = x.reshape(x.shape[0], -1)
            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
            for result in predictions:
                predictions_csv["target"].append(int(result))
                predictions_csv["ID"].append(count)
                count+=1
    df = pd.DataFrame(predictions_csv)
    df.to_csv("submission.csv", index=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

train_dataset=ExtendedMNISTDataset(root="/Users/alexrimniceanu/facultate/anul3/sem1/nn/Neural-Networks/homework4", train=True)
test_dataset=ExtendedMNISTDataset(root="/Users/alexrimniceanu/facultate/anul3/sem1/nn/Neural-Networks/homework4", train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,8], gamma=0.1)
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)
        
        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
 
        # Backward
        optimizer.zero_grad()
        loss.backward()
 
        # Gradient descent
        optimizer.step()
    scheduler.step()
solve(test_loader, model)