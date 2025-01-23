# model_saver.py

import pickle

def save_model(model, filename):
    """
    Saves the trained model to a pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    from model_builder import build_and_train_model
    from data_preprocessing import preprocess_data
    from wine_data_loader import load_wine_data

    # Load and preprocess the data
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train the model
    model = build_and_train_model(X_train, y_train)

    # Save the model
    save_model(model, 'wine_quality_model.pkl')
