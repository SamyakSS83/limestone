import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
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

# def bust_probability(dealer_cards):
#     """
#     dealer_cards: list -> integer series of player denoting value of cards observed upto this point

#     Current body is random for now, change it accordingly
    
#     output: probability of going bust on this table
#     """
#     bust_count = 0
#     total_count = 0
#     n = len(dealer_cards)
#     for i in range(n):
#         index  = i
#         total = 0

#         while(total <= 16 and index < n):
#             total += dealer_cards[index]
#             index += 1
#         if(total>21):
#             bust_count += 1
#         total_count += 1
#     if total_count != 0:
#         return bust_count/total_count
#     else:   


#         return 0
    
def bust_probability(dealer_cards):
    """
    dealer_cards: list of integers representing card values (2-10, 10 for face cards, 11 for aces)
    
    output: probability of dealer busting based on sequential hand simulation
    """
    bust_count = 0
    total_count = 0
    index = 0
    n = len(dealer_cards)
    
    while index < n:
        total = 0
        
        while total <= 16 and index < n:
            total += dealer_cards[index]
            index += 1
        if total > 21:
            bust_count += 1
        total_count += 1
    
    return bust_count / total_count if total_count > 0 else 0


data = data_loader("train.csv", 0, 'player')  
spy_values = data[:, 0]
card_values = data[:, 1]

plt.scatter(spy_values, card_values, alpha=0.5)
plt.xlabel('Spy Value')
plt.ylabel('Card Value')
plt.title('Spy Values vs Card Values')
plt.grid(True)
plt.show()
