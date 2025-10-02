import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df: pd.DataFrame):
    """Runs and saves all exploratory data analysis plots."""
    print("Running Exploratory Data Analysis...")

    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Гүйлгээний төрөл тархалт (0: Луйвар || 1: Хэвийн)')
    plt.xlabel("Гүйлгээний төрөл")
    plt.ylabel("Давтамж")
    plt.savefig("images/01_class_distribution.png")
    plt.close()

    # Time and Amount distributions
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    sns.histplot(df['Amount'], ax=ax[0], color='r', kde=True)
    ax[0].set_title('Гүйлгээний дүнгийн тархалт')
    sns.histplot(df['Time'], ax=ax[1], color='b', kde=True)
    ax[1].set_title('Гүйлгээний цагийн тархалт')
    plt.savefig("images/02_scaled_distributions.png")
    plt.close()

    # Correlation matrix on an undersampled subset for better visualization
    fraud_df = df[df['Class'] == 1]
    non_fraud_df = df[df['Class'] == 0].sample(n=len(fraud_df), random_state=42)
    sub_sample = pd.concat([fraud_df, non_fraud_df])
    
    plt.figure(figsize=(24, 20))
    sub_sample_corr = sub_sample.corr()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20})
    plt.title('Subsample Correlation Matrix')
    plt.savefig("images/03_correlations.png")
    plt.close()

    # Kernel PCA features boxplots
    pca_features = [f'V{i}' for i in range(1, 29)]
    n_cols = 4
    n_rows = -(-len(pca_features) // n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, n_rows*3))
    axes = axes.flatten()
    colors = ["#0101DF", "#DF0101"]
    for i, feature in enumerate(pca_features):
        sns.boxplot(x="Class", y=feature, data=df, palette=colors, ax=axes[i])
        axes[i].set_title(f'{feature}')
    for j in range(len(pca_features), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    plt.savefig("images/04_boxplots.png")
    plt.close()

    print("EDA plots saved to current directory.")