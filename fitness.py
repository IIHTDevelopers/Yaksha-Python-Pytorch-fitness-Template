import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def bin_calories(y):
    """
    Bin calories into 3 categories:
    - 0: 0-200 calories (low)
    - 1: 200-350 calories (medium)
    - 2: 350-1000 calories (high)

    Args:
        y: pandas Series or array-like of calorie values

    Returns:
        pandas Series of integer labels (0, 1, 2)
    """
    # TODO: Implement calorie binning using pd.cut
    # Use bins=[0, 200, 350, 1000] and labels=[0, 1, 2]
    # Convert to int type and return
    return pd.Series([0] * len(y))  # Placeholder return


def load_data_from_csv(path='fitness_data.csv'):
    """
    Load fitness data from CSV file and prepare it for training.

    Steps:
    1. Load CSV data using pandas
    2. Bin the calories_burned column using bin_calories function
    3. Handle any null values in binned data
    4. Separate features (X) and target (y)
    5. Scale features using StandardScaler
    6. Split data using StratifiedShuffleSplit (80/20 split)
    7. Convert to PyTorch tensors

    Args:
        path: Path to the CSV file

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as PyTorch tensors
    """
    # TODO: Load CSV data using pd.read_csv

    # TODO: Bin calories using bin_calories function

    # TODO: Handle null values - remove rows where binning failed

    # TODO: Separate features (drop 'calories_burned' column) and target

    # TODO: Scale features using StandardScaler

    # TODO: Use StratifiedShuffleSplit for train/test split
    # Use n_splits=1, test_size=0.2, random_state=42

    # TODO: Convert to PyTorch tensors
    # X tensors should be float32, y tensors should be long

    # Placeholder return values to prevent errors
    X_train = torch.empty((0, 5), dtype=torch.float32)  #---> chnage the value according to you needed dummy values provied
    y_train = torch.empty((0,), dtype=torch.long)       #---> chnage the value according to you needed dummy values provied
    X_test = torch.empty((0, 5), dtype=torch.float32)   #---> chnage the value according to you needed dummy values provied
    y_test = torch.empty((0,), dtype=torch.long)        #---> chnage the value according to you needed dummy values provied
    return X_train, y_train, X_test, y_test


class FitnessDataset(Dataset):
    """
    Custom PyTorch Dataset for fitness data.

    This class should inherit from torch.utils.data.Dataset and implement
    the required methods: __init__, __len__, and __getitem__.
    """

    def __init__(self, X, y):
        """
        Initialize the dataset with features and labels.

        Args:
            X: Feature tensor
            y: Label tensor
        """
        # TODO: Store X and y as instance variables
        self.X = torch.randn(10, 5)  # Placeholder
        self.y = torch.zeros(10, dtype=torch.long)  # Placeholder

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        # TODO: Return the length of the dataset
        return 10  # Placeholder return

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            tuple: (features, label) for the given index
        """
        # TODO: Return the sample at the given index as a tuple (X[idx], y[idx])
        return torch.randn(5), torch.tensor(0, dtype=torch.long)  # Placeholder return


def build_model(input_size=5, num_classes=3):
    """
    Build a neural network model for fitness classification.

    Architecture should include:
    - Input layer (Linear): input_size -> 16
    - ReLU activation
    - Dropout layer (0.3 dropout rate)
    - Hidden layer (Linear): 16 -> 8
    - ReLU activation
    - Output layer (Linear): 8 -> num_classes

    Args:
        input_size: Number of input features (default: 5)
        num_classes: Number of output classes (default: 3)

    Returns:
        nn.Sequential: The constructed model
    """
    # TODO: Create a sequential model with the specified architecture
    # Use nn.Sequential, nn.Linear, nn.ReLU, nn.Dropout
    return nn.Linear(input_size, num_classes)  # Placeholder simple model


def train_model(model, dataloader, val_loader=None, epochs=15, lr=0.01):
    """
    Train the neural network model.

    Steps:
    1. Set up loss function (CrossEntropyLoss)
    2. Set up optimizer (Adam)
    3. Training loop for specified epochs
    4. For each epoch: forward pass, compute loss, backward pass, update weights
    5. Optional: print validation accuracy if val_loader provided

    Args:
        model: The neural network model to train
        dataloader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs (default: 15)
        lr: Learning rate (default: 0.01)
    """
    # TODO: Set up CrossEntropyLoss as criterion

    # TODO: Set up Adam optimizer with given learning rate

    # TODO: Implement training loop
    # For each epoch:
    #   - Set model to training mode
    #   - For each batch: zero gradients, forward pass, compute loss, backward pass, step
    #   - Optionally evaluate on validation set and print accuracy

    pass


def evaluate_model(model, dataloader):
    """
    Evaluate the model on a dataset.

    Steps:
    1. Set model to evaluation mode
    2. Disable gradient computation
    3. For each batch: forward pass, get predictions
    4. Calculate accuracy

    Args:
        model: The trained model
        dataloader: Data loader for evaluation

    Returns:
        float: Accuracy as a decimal (e.g., 0.85 for 85%)
    """
    # TODO: Set model to evaluation mode

    # TODO: Initialize counters for correct predictions and total samples

    # TODO: Disable gradient computation using torch.no_grad()

    # TODO: For each batch:
    #   - Get model outputs
    #   - Get predicted classes using torch.max
    #   - Count correct predictions
    #   - Update total count

    # TODO: Return accuracy (correct / total)

    return 0


def save_model(model, path='fitness_model_class.pth'):
    """
    Save the model's state dictionary to a file.

    Args:
        model: The trained model
        path: Path to save the model (default: 'fitness_model_class.pth')
    """
    # TODO: Save model state dict using torch.save
    pass


def load_model(path='fitness_model_class.pth', input_size=5, num_classes=3):
    """
    Load a saved model from file.

    Steps:
    1. Create a new model with build_model
    2. Load the state dictionary
    3. Set model to evaluation mode

    Args:
        path: Path to the saved model
        input_size: Number of input features
        num_classes: Number of output classes

    Returns:
        nn.Sequential: The loaded model
    """
    # TODO: Create a new model using build_model

    # TODO: Load state dictionary using torch.load

    # TODO: Set model to evaluation mode

    # TODO: Return the loaded model

    return nn.Linear(input_size, num_classes)  # Placeholder return


if __name__ == "__main__":
    """
    Main execution block for training and evaluating the fitness model.
    
    This block should:
    1. Load data from CSV
    2. Create datasets and data loaders
    3. Build the model
    4. Train the model
    5. Evaluate the model
    6. Save the trained model
    """
    # TODO: Load data using load_data_from_csv

    # TODO: Create FitnessDataset instances for train and test data

    # TODO: Create DataLoader instances (batch_size=4, shuffle=True for training)

    # TODO: Build model with correct input size

    # TODO: Train the model

    # TODO: Evaluate the model and print accuracy

    # TODO: Save the trained model

    pass
