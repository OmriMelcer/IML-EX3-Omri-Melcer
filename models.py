import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        ########## YOUR CODE HERE ##########

        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv
        n_train = X.shape[0]
        # Formula: W* = (X^T X / n_train + lambda I)^-1 (X^T Y / n_train)
        # We use np.linalg.pinv (Pseudo-inverse, SVD based) to ensure the inner part is invertible.
        inner = (X.T @ X) / n_train + self.lambd * np.eye(X.shape[1])
        self.weights = np.linalg.pinv(inner) @ (X.T @ Y) / n_train
        

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. 
        :return: The predicted output. 
        """
        # Add bias term to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        preds = X @ self.weights
        preds = np.where(preds >= 0, 1, -1)
        return preds



class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.
        self.linear = nn.Linear(input_dim, output_dim)
        ####################################
        

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.
        return self.linear(x)
        
        

        

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x

class SimpleSet(Dataset):
    def __init__(self,X,y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(y).long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
        
        
        
        