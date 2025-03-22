import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd

def data_loader(path, table_idx, player_or_dealer):
    #utility for loading train.csv, example use in the notebook
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

class MyPlayer:
    def __init__(self,table_index):
        pass  
        
    def get_card_value_from_spy_value(self,value):
        """
        value: a value from the spy series as a float
        
        It is the same function you found in the previous part
        We will not judge this function in this part, so you can choose the return type as you prefer.
        Only make sure you return the correct value as you will be using this function
        
        The body is random  for now, rewrite accordingly

        Output:
            return a scalar value of the prediction
        """
        return 10
        
    def get_player_spy_prediction(self,hist):
        """
        hist a 1D numpy array of size (len_history,) len_history=5
        return a scalar value of the prediction

        The body is random  for now, rewrite accordingly

        Output:
            return a scalar value of the prediction
        """

        return 1e6

    def get_dealer_spy_prediction(self,hist):
        """
        hist a 1D numpy array of size (len_history,) len_history=5
        
        The body is random  for now, rewrite accordingly

        Output:
            return a scalar value of the prediction
        """

        return 1e6