# model_builder.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def build_and_train_model(X_train, y_train):
    """
    Builds and trains a RandomForestClassifier model.
    """
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from wine_data_loader import load_wine_data

    # Load and preprocess the data
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train the model
    model = build_and_train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
