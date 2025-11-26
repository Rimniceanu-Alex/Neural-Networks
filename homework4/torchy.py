import torch
import torchvision.datasets as datasets # for Mist
import torchvision.transforms as transforms # Transformations we can perform on our dataset for augmentation
from torch import optim # For optimizers like SGD, Adam, etc.
from torch import nn # To inherit our neural network
from torch.utils.data import DataLoader # For management of the dataset (batches)
from tqdm import tqdm # For nice progress bar!

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

def check_accuracy(loader, model):
 
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)
 
            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
    
            # Check how many we got correct
            num_correct += (predictions == y).sum()
    
            # Keep track of number of samples
            num_samples += predictions.size(0)
    model.train()
    return num_correct / num_samples



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 20
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,8], gamma=0.1)
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Try to move to GPU
        data = data.to(device=device)
        targets = targets.to(device=device)
 
        # Get to correct shape
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
print(f"Accuracy: {check_accuracy(test_loader, model)}")
