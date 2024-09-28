import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid', init_method='xavier'):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation function")
        
        self.init_weights(init_method)
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x
    
    def init_weights(self, method):
        if method == 'zeros':
            nn.init.zeros_(self.hidden.weight)
            nn.init.zeros_(self.output.weight)
        elif method == 'ones':
            nn.init.ones_(self.hidden.weight)
            nn.init.ones_(self.output.weight)
        elif method == 'uniform':
            nn.init.uniform_(self.hidden.weight, -1/np.sqrt(self.hidden.in_features), 1/np.sqrt(self.hidden.in_features))
            nn.init.uniform_(self.output.weight, -1/np.sqrt(self.output.in_features), 1/np.sqrt(self.output.in_features))
        elif method == 'normal':
            nn.init.normal_(self.hidden.weight, std=0.01)
            nn.init.normal_(self.output.weight, std=0.01)
        elif method == 'xavier':
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.xavier_uniform_(self.output.weight)
        elif method == 'he':
            nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.output.weight, nonlinearity='relu')
        elif method == 'uniform_scaled':
            self._init_uniform_scaled(self.hidden)
            self._init_uniform_scaled(self.output)
        else:
            raise ValueError("Unsupported initialization method")

    def _init_uniform_scaled(self, layer):
        n = layer.in_features
        y = 1.0 / np.sqrt(n)
        nn.init.uniform_(layer.weight, -y, y)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def prepare_data(X, y, batch_size=32):
    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, (X_test_tensor, y_test_tensor)

def train_model(model, train_loader, test_data, criterion, optimizer, epochs=100):
    X_test, y_test = test_data
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            losses.append(test_loss.item())
            
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).float().mean()
            accuracies.append(accuracy.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    
    return losses, accuracies

def plot_results(losses, accuracies, init_method):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title(f'Loss vs Epochs ({init_method})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    ax2.plot(accuracies)
    ax2.set_title(f'Accuracy vs Epochs ({init_method})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate some dummy data for demonstration
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    input_size = X.shape[1]
    hidden_size = 64
    output_size = 2
    learning_rate = 0.01
    epochs = 100
    
    train_loader, test_data = prepare_data(X, y)
    
    init_methods = ['zeros', 'ones', 'uniform', 'normal', 'xavier', 'he', 'uniform_scaled']
    
    for init_method in init_methods:
        print(f"\nTraining with {init_method} initialization:")
        model = NeuralNetwork(input_size, hidden_size, output_size, init_method=init_method)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        losses, accuracies = train_model(model, train_loader, test_data, criterion, optimizer, epochs)
        plot_results(losses, accuracies, init_method)

if __name__ == "__main__":
    main()
