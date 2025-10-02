import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, auc

import numpy as np

def calculate_profit(y_true, y_pred, amounts, txn_fee):
    """
    Profit rules:
    - Correctly allow normal txn (y_true=0, y_pred=0) → +txn_fee * amount
    - Missed fraud (y_true=1, y_pred=0) → -amount
    - Blocked txns (y_pred=1) → 0
    """
    profit = 0.0
    
    # Case 1: true normal & predicted normal → earn fee
    profit += np.sum(amounts[(y_true == 0) & (y_pred == 0)]) * txn_fee
    
    # Case 2: fraud but predicted normal → lose amount
    profit -= np.sum(amounts[(y_true == 1) & (y_pred == 0)])
    
    return profit

def train_and_evaluate_models(X_train, y_train, X_test, y_test, M_test):
    """
    Trains a suite of classifiers and evaluates them on the test set.
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42)
    }
    TXN_FEE = 0.02 # Transaction fee percentage
    
    results = {}
    trained_models = {}

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, target_names=['No Fraud', 'Fraud'], output_dict=True)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        profit = calculate_profit(y_test.values, y_pred, M_test, TXN_FEE)
        
        results[name] = {
            "Precision": report['Fraud']['precision'],
            "Recall": report['Fraud']['recall'],
            "F1-Score": report['Fraud']['f1-score'],
            "AUC-ROC": roc_auc_score(y_test, y_proba),
            "AUC-PR": auc(recall, precision),
            "Profit": profit
        }
        
        trained_models[name] = clf
        print(f"--- {name} Results ---")
        print(classification_report(y_test, y_pred, target_names=['No Fraud', 'Fraud']))
        print(f"AUC-ROC: {results[name]['AUC-ROC']:.4f}")
        print(f"AUC-PR: {results[name]['AUC-PR']:.4f}\n")
        print(f"Profit: ${profit:.2f}\n")
    return results, trained_models