# main.py

from wine_data_loader import load_wine_data
from data_preprocessing import preprocess_data
from model_builder import build_and_train_model, evaluate_model
from model_saver import save_model
from model_predictor import load_model, make_predictions

def main():
    # Step 1: Load the dataset
    print("Loading the wine dataset...")
    X, y = load_wine_data()

    # Step 2: Preprocess the data
    print("Preprocessing the data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Build and train the model
    print("Building and training the model...")
    model = build_and_train_model(X_train, y_train)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Step 5: Save the model
    print("Saving the model...")
    save_model(model, 'wine_quality_model.pkl')

    # Step 6: Load the model and make predictions
    print("Loading the model and making predictions...")
    model_loaded = load_model('wine_quality_model.pkl')
    predictions = make_predictions(model_loaded, X_test)
    print(f"Predictions: {predictions[:10]}")

if __name__ == "__main__":
    main()
