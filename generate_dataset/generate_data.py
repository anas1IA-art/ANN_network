import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd

__all__ = ["SimpleDataset", ]


class SimpleDataset:
    def __init__(self, n_samples=1000, noise=0.1):
        self.n_samples = n_samples
        self.noise = noise
        self.X, self.y = self.generate_data()

    def generate_data(self):
        # Generate two-dimensional input data
        X = np.random.randn(self.n_samples, 2)
        
        # Generate target values
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Add some noise to make it more challenging
        X += np.random.randn(self.n_samples, 2) * self.noise
        
        return X, y

    def get_train_test_split(self, test_size=0.2):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def plot_data(self):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], label='Class 0', alpha=0.5)
        plt.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], label='Class 1', alpha=0.5)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Simple Classification Dataset')
        plt.legend()
        plt.show()


# if __name__ == "__main__":
#     # Create the dataset
#     dataset = SimpleDataset(n_samples=1000, noise=0.1)

#     X,y = dataset.generate_data()

#     data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

#     data['label'] =  y
   
#     data.to_csv("genrate/dataset.csv", index= False)