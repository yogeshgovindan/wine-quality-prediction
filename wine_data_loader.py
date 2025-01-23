# wine_data_loader.py

from ucimlrepo import fetch_ucirepo

def load_wine_data():
    """
    Loads the Wine Quality dataset from the UCIML repository.
    """
    # Fetch the dataset
    wine_quality = fetch_ucirepo(id=186)

    # Extract features (X) and target (y)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # Return the dataset
    return X, y

if __name__ == "__main__":
    # Load the data
    X, y = load_wine_data()
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
