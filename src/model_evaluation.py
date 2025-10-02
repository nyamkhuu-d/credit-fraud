import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curves(models: dict, strategy_name: str, X_test, y_test):
    """
    Plots the Precision-Recall curve for each trained model.
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)
            plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for All Models ({strategy_name})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'evaluation_pr_curve_{strategy_name.lower()}.png')
    plt.close()