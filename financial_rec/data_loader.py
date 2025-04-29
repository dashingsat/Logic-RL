import json
import random
from typing import List, Dict, Tuple, Any

def load_synthetic_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads synthetic financial reconciliation data from a JSON file.

    Args:
        file_path: The path to the JSON file containing the data.
                   Expected format is a list of dictionaries.

    Returns:
        A list of dictionaries, where each dictionary represents a data sample.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file does not contain a list.")
        print(f"Successfully loaded {len(data)} samples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        raise

def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        data: The list of data samples (dictionaries).
        train_ratio: The proportion of data to use for the training set.
        val_ratio: The proportion of data to use for the validation set.
        test_ratio: The proportion of data to use for the test set.
        shuffle: Whether to shuffle the data before splitting.
        random_seed: The random seed to use for shuffling (for reproducibility).

    Returns:
        A tuple containing the training, validation, and test datasets (lists of samples).
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    if shuffle:
        random.seed(random_seed)
        shuffled_data = random.sample(data, len(data)) # Create a shuffled copy
    else:
        shuffled_data = data # Use original order if not shuffling

    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test is the remainder to ensure all data is used

    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train : n_train + n_val]
    test_data = shuffled_data[n_train + n_val :]

    print(f"Data split complete:")
    print(f"  Training set size: {len(train_data)}")
    print(f"  Validation set size: {len(val_data)}")
    print(f"  Test set size: {len(test_data)}")

    return train_data, val_data, test_data

# Example usage (optional, can be run if the script is executed directly)
if __name__ == '__main__':
    # Assuming synthetic_data.json is in the financial_rec/data/ directory 
    # relative to the workspace root when running with 'python -m'
    data_path = 'financial_rec/data/synthetic_data.json' 
    
    try:
        all_data = load_synthetic_data(data_path)
        
        # Example: Splitting the data
        train_set, val_set, test_set = split_data(all_data)
        
        # You can now use train_set, val_set, test_set for further processing
        print(f"First sample from training set: {train_set[0] if train_set else 'Empty'}")
        
    except Exception as e:
        print(f"An error occurred during example usage: {e}") 