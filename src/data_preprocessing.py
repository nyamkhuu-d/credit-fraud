import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales 'Time' and 'Amount' columns using RobustScaler."""
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE to handle class imbalance."""
    print("Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    print(f"Original training shape: {X_train.shape}")
    print(f"Shape after SMOTE: {X_smote.shape}")
    return X_smote, y_smote

def create_undersample(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Creates a balanced dataset by undersampling the majority class."""
    print("Creating a random undersample...")
    df = pd.concat([X_train, y_train], axis=1)
    fraud_df = df[df['Class'] == 1]
    non_fraud_df = df[df['Class'] == 0].sample(n=len(fraud_df), random_state=42)
    
    undersampled_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)
    
    X_under = undersampled_df.drop('Class', axis=1)
    y_under = undersampled_df['Class']
    
    print(f"Undersampled training shape: {X_under.shape}")
    return X_under, y_under