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

def create_clustering_plot():
    # Load data for table 0
    player_data = data_loader('train.csv', 0, 'player')
    dealer_data = data_loader('train.csv', 0, 'dealer')
    
    # Combine player and dealer data
    all_data = np.vstack([player_data, dealer_data])
    
    # Create labels for the plot
    labels = ['Player'] * len(player_data) + ['Dealer'] * len(dealer_data)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(all_data[:, 0], all_data[:, 1], c=[0 if l == 'Player' else 1 for l in labels], 
                            alpha=0.6, cmap='viridis')
    
    # Add labels and title
    plt.xlabel('Spy Value')
    plt.ylabel('Card Value')
    plt.title('Clustering of Spy vs Card Values for Table 0')
    
    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=['Player', 'Dealer'])
    
    # Show the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Player/Dealer')
    plt.show()

def create_separate_plots(x):
    # Load data for table 0
    player_data = data_loader('train.csv', x, 'player')
    dealer_data = data_loader('train.csv', x, 'dealer')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Player plot
    scatter_player = ax1.scatter(player_data[:, 0], player_data[:, 1], 
                                 alpha=0.7, c='blue', label='Player')
    ax1.set_xlabel('Spy Value')
    ax1.set_ylabel('Card Value')
    ax1.set_title('Player: Spy vs Card Values for Table {}'.format(x))
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Dealer plot
    scatter_dealer = ax2.scatter(dealer_data[:, 0], dealer_data[:, 1], 
                                alpha=0.7, c='green', label='Dealer')
    ax2.set_xlabel('Spy Value')
    ax2.set_ylabel('Card Value')
    ax2.set_title('Dealer: Spy vs Card Values for Table {}'.format(x))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def create_clustering_models():
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import silhouette_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import pickle
    import os
    
    # Create directory for saving models if it doesn't exist
    os.makedirs('clustering_models', exist_ok=True)
    
    results = []
    
    # Process each table
    for table_idx in range(5):
        # Process player and dealer for each table
        for role in ['player', 'dealer']:
            print(f"Training clustering model for table {table_idx}, {role}")
            
            # Load data
            data = data_loader('train.csv', table_idx, role)
            
            # Split into train and test sets (80% train, 20% test)
            X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Determine optimal number of clusters using silhouette score
            best_score = -1
            best_k = 2
            for k in range(2, 41):  # Try 2-10 clusters
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_train_scaled)
                score = silhouette_score(X_train_scaled, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Train final model with optimal number of clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            kmeans.fit(X_train_scaled)
            
            # Evaluate model
            # For visualization purposes, predict cluster centers for test data
            cluster_labels = kmeans.predict(X_test_scaled)
            centroids = kmeans.cluster_centers_
            
            # Predict card values by assigning the card value of the closest centroid
            predicted_cards = np.zeros(len(X_test))
            for i, label in enumerate(cluster_labels):
                predicted_cards[i] = centroids[label][1]  # Card value is at index 1
            
            # Calculate error metrics
            mse = mean_squared_error(X_test[:, 1], predicted_cards)
            
            # Save model and scaler
            model_filename = f'clustering_models/model_table{table_idx}_{role}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump({
                    'kmeans': kmeans,
                    'scaler': scaler,
                    'best_k': best_k,
                    'silhouette_score': best_score
                }, f)
            
            # Store results for reporting
            results.append({
                'table': table_idx,
                'role': role,
                'clusters': best_k,
                'silhouette_score': best_score,
                'mse': mse
            })
            
            # Visualization of clusters
            plt.figure(figsize=(12, 6))
            
            # Plot test data colored by cluster
            plt.subplot(1, 2, 1)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=30, alpha=0.7)
            plt.scatter(scaler.inverse_transform(centroids)[:, 0], scaler.inverse_transform(centroids)[:, 1], 
                      c='red', marker='X', s=100)
            plt.title(f'Clusters for Table {table_idx} {role.capitalize()}')
            plt.xlabel('Spy Value')
            plt.ylabel('Card Value')
            
            # Plot predicted vs actual
            plt.subplot(1, 2, 2)
            plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', marker='o', s=30, alpha=0.4, label='Actual')
            plt.scatter(X_test[:, 0], predicted_cards, c='red', marker='x', s=30, alpha=0.4, label='Predicted')
            plt.title(f'Prediction for Table {table_idx} {role.capitalize()}\nMSE: {mse:.4f}')
            plt.xlabel('Spy Value')
            plt.ylabel('Card Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'clustering_models/plot_table{table_idx}_{role}.png')
            plt.close()
    
    # Display summary of results
    results_df = pd.DataFrame(results)
    print("\nClustering Model Results:")
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv('clustering_models/results_summary.csv', index=False)
    
    return results_df

def predict_with_clustering_models(table_idx, role, spy_values):
    """
    Predict card values using the trained clustering model
    
    Parameters:
    table_idx (int): Table index (0-4)
    role (str): 'player' or 'dealer'
    spy_values (array): Array of spy values to predict card values for
    
    Returns:
    array: Predicted card values
    """
    import pickle
    
    # Load the model
    model_filename = f'clustering_models/model_table{table_idx}_{role}.pkl'
    try:
        with open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        kmeans = model_data['kmeans']
        scaler = model_data['scaler']
        
        # Prepare the input data (add dummy card values that will be ignored)
        dummy_data = np.column_stack((spy_values, np.zeros(len(spy_values))))
        
        # Scale the data
        scaled_data = scaler.transform(dummy_data)
        
        # Predict clusters
        clusters = kmeans.predict(scaled_data)
        
        # Get centroids
        centroids = kmeans.cluster_centers_
        
        # Extract card values from centroids based on predicted clusters
        predicted_cards = np.array([centroids[cluster][1] for cluster in clusters])
        
        # Convert back to original scale
        predicted_data = np.column_stack((spy_values, predicted_cards))
        original_scale = scaler.inverse_transform(np.column_stack((scaled_data[:, 0], predicted_cards)))
        predicted_cards = original_scale[:, 1]
        
        return predicted_cards
    
    except FileNotFoundError:
        print(f"Model file not found: {model_filename}")
        return None

# Example of how to use the prediction function
def test_prediction_example():
    table_idx = 0
    role = 'player'
    # Sample spy values to predict card values for
    spy_values = np.array([0.3, 0.5, 0.7, 0.9])
    
    predicted_cards = predict_with_clustering_models(table_idx, role, spy_values)
    
    if predicted_cards is not None:
        for spy, card in zip(spy_values, predicted_cards):
            print(f"Spy: {spy:.2f}, Predicted Card: {card:.2f}")

# Call the function to create and display the plot
# create_clustering_plot()

# Call the new function to create and display separate plots
create_separate_plots(0)
create_separate_plots(1)
create_separate_plots(2)
create_separate_plots(3)
create_separate_plots(4)

# Uncomment the following line to train and save all clustering models
# results = create_clustering_models()

# Uncomment the following line to test the prediction function
# test_prediction_example()


