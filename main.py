import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    scale_features,
    apply_smote,
    create_undersample,
)
from src.eda import run_eda
from src.model_training import train_and_evaluate_models
from src.model_evaluation import plot_precision_recall_curves

def main():
    try:
        data_df = pd.read_csv("data/creditcard.csv")
        # Amount for profit calculation
        M = data_df['Amount']
    except FileNotFoundError:
        print("Error: 'data/creditcard.csv' not found.")
        print("Please download the dataset from Kaggle and place it in the 'data' directory.")
        print("Dataset URL: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        return

    # --- 1. Exploratory Data Analysis ---
    run_eda(data_df)

    # --- 2. Data Preprocessing ---
    # Scale 'Amount' and 'Time' features
    data_df = scale_features(data_df)

    # Define predictors and target
    predictors = [col for col in data_df.columns if col not in [target, 'Time', 'Amount']]
    target = 'Class'
    
    # Create the train-test split for final evaluation
    X = data_df[predictors]
    y = data_df[target]
    X_train_orig, X_test, y_train_orig, y_test, M_train, M_test = train_test_split(
        X, y, M, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Original Data ---
    print("="*50)
    print("Running Analysis with Original Imbalanced Data")
    print("="*50)
    
    # Train models on the original, imbalanced training data
    results_orig, models_orig = train_and_evaluate_models(
        X_train_orig, y_train_orig, X_test, y_test, M_test
    )
    plot_precision_recall_curves(models_orig, "Original", X_test, y_test)


    # --- 4. Undersampling ---
    print("\n" + "="*50)
    print("Running Analysis with Undersampling Strategy")
    print("="*50)
    
    # Create a balanced undersample from the original training data
    X_train_under, y_train_under = create_undersample(X_train_orig, y_train_orig)
    
    # Train models on the processed undersampled data
    results_under, models_under = train_and_evaluate_models(
        X_train_under, y_train_under, X_test, y_test, M_test
    )
    plot_precision_recall_curves(models_under, "Undersampling", X_test, y_test)


    # --- 5. SMOTE Strategy ---
    print("\n" + "="*50)
    print("Running Analysis with SMOTE Oversampling Strategy")
    print("="*50)

    # Apply SMOTE to the original training data
    X_train_smote, y_train_smote = apply_smote(X_train_orig, y_train_orig)
    
    # Train models on the SMOTE-resampled data
    results_smote, models_smote = train_and_evaluate_models(
        X_train_smote, y_train_smote, X_test, y_test, M_test
    )
    plot_precision_recall_curves(models_smote, "SMOTE", X_test, y_test)

if __name__ == "__main__":
    main()