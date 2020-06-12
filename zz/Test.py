
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
# Linear model f = wx + b
input_size = n_features
output_size = 1

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 5, bias = False)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(5, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, output_size)
model.l2.bias


# model = nn.Linear(input_size, output_size)


# 2) Loss and optimizer
learning_rate = 0.01

criterion = nn.MSELoss()
print(model.l1.weight)
print(model.l1.bias)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 2
for epoch in range(num_epochs):
    # Forward pass and loss
    if(epoch ==0):
        model.l1.weight.requires_grad=True
        model.l1.bias.requires_grad=False
    if(epoch ==1):
        model.l1.weight.requires_grad=True
        model.l1.bias.requires_grad=True

    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    print(model.l1.weight)
    print(model.l1.bias)

# Plot
predicted = model(X).detach().numpy()
# predicted
