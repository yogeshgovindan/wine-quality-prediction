# model_predictor.py

import pickle

def load_model(filename):
    """
    Loads the trained model from the pickle file.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(model, X):
    """
    Makes predictions using the trained model.
    """
    return model.predict(X)

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from wine_data_loader import load_wine_data

    # Load and preprocess the data
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Load the saved model
    model = load_model('wine_quality_model.pkl')

    # Make predictions on the test data
    predictions = make_predictions(model, X_test)
    print(f"Predictions: {predictions[:10]}")
