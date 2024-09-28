import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split



__all__ = ['AnnTrainer',]

class AnnTrainer:
    def __init__(self, dataset, batch_size=32, test_size=0.2):

        # super(AnnTrainer, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_loader, self.test_loader = self.create_data_loaders()

    def create_data_loaders(self):
        # Split the data
        train_indices, test_indices = train_test_split(
            range(len(self.dataset)), test_size=self.test_size, random_state=42
        )

        # Create subsets
        train_dataset = Subset(self.dataset, train_indices)
        test_dataset = Subset(self.dataset, test_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    
    # def train_step(self)

    def train_model(self, model, criterion, optimizer, epochs=100):
        train_losses = []
        losses = []
        accuracies = []

        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss/ len(self.train_loader)
            train_losses.append(avg_train_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch train result [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}')


            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in self.test_loader:
                    outputs = model(batch_X)
                    
                    test_loss += criterion(outputs.squeeze(1), batch_y.float()).item()
                   
                    predicted = outputs.round() 
                   
                    total += batch_y.size(0)
                    correct += (predicted.squeeze(1) == batch_y.float()).sum().item()

            avg_loss = test_loss / len(self.test_loader)
            accuracy = correct / total
            losses.append(avg_loss)
            accuracies.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch test result [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        return train_losses , losses, accuracies

  

