# data_preprocessing.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, y):
    """
    Preprocess the wine dataset by splitting into train and test sets,
    and scaling the features.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    from wine_data_loader import load_wine_data

    # Load the data
    X, y = load_wine_data()

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Processed X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
