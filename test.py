from generate_dataset.generate_data import SimpleDataset
import torch.nn as nn 
import torch.optim as optim
import pandas  as pd
from plots.plot_results import ResultPlotter
import argparse

from datasets.preprocessing import PrepaDataset

from models.base_model import NeuralNetwork

from train_models.train_model import AnnTrainer

dataset = SimpleDataset(n_samples=1000, noise=0.1)

X,y = dataset.generate_data()

data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

data['label'] =  y

data.to_csv("./data/data.csv", index= False)


pre = PrepaDataset("data/data.csv")

      


def main(arg):

    model =  NeuralNetwork(input_size=2,hidden_size=14,output_size=1,init_method=arg.init_method)
    
    Train = AnnTrainer(pre)

    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
     
    train_losses , losses, accuracies = Train.train_model(model= model ,criterion= criterion , optimizer= optimizer ,epochs=arg.epochs)

    track_resluts = ResultPlotter(arg.init_method)
    track_resluts.plot_results(losses= losses , accuracies=accuracies)
    track_resluts.save_plot("plots/results")    



parser = argparse.ArgumentParser(description="Train a neural network model.")



parser.add_argument("--init_method", type=str, default="normal",
                    help="initializarion method")
parser.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs for training")



args = parser.parse_args()

main(args)