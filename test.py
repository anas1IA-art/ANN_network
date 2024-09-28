from generate_dataset.generate_data import SimpleDataset
import torch.nn as nn 
import torch.optim as optim
import pandas  as pd

from datasets.preprocessing import PrepaDataset

from models.base_model import NeuralNetwork

from train_models.train_model import AnnTrainer

dataset = SimpleDataset(n_samples=1000, noise=0.1)

X,y = dataset.generate_data()

data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

data['label'] =  y

data.to_csv("./data/data.csv", index= False)


pre = PrepaDataset("data/data.csv")

      
model =  NeuralNetwork(input_size=2,hidden_size=14,output_size=1,init_method='uniform_scaled')

Train = AnnTrainer(pre)
# print(next(iter(Train.train_loader))[0])

criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

Train.train_model(model= model ,criterion= criterion , optimizer= optimizer ,epochs=100)