# src/data_prep.py
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, '..', 'data'))
DATA_PATH = os.path.join(DATA_DIR, 'credit_data.csv')
RESULTS_DIR = os.path.normpath(os.path.join(HERE, '..', 'results'))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_synthetic_data(n_samples=3000, random_state=42):
    """
    Generate synthetic credit dataset for model demonstration.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.72, 0.28],  # typical class imbalance in credit scoring
        class_sep=1.5,
        flip_y=0.02,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[
        "age", "income", "loan_amount", "loan_duration", "num_dependents",
        "prev_defaults", "employment_years", "credit_utilization"
    ])
    # Apply realistic value scaling
    df["age"] = np.round(20 + np.abs(df["age"] * 12)).astype(int)
    df["income"] = np.round(20000 + np.abs(df["income"] * 14000)).astype(int)
    df["loan_amount"] = np.round(1000 + np.abs(df["loan_amount"] * 2500)).astype(int)
    df["loan_duration"] = np.round(6 + np.abs(df["loan_duration"] * 8)).astype(int)
    df["num_dependents"] = np.clip(np.round(np.abs(df["num_dependents"] * 1.5)), 0, 5).astype(int)
    df["prev_defaults"] = np.clip(np.round(np.abs(df["prev_defaults"])), 0, 3).astype(int)
    df["employment_years"] = np.clip(np.round(1 + np.abs(df["employment_years"] * 3)), 0, 35).astype(int)
    df["credit_utilization"] = np.clip(np.abs(df["credit_utilization"]) * 0.7, 0, 1)
    df['default'] = y
    return df

def ensure_data_exists():
    """
    Check if data/credit_data.csv exists. If not, generate synthetic data and save.
    """
    if not os.path.exists(DATA_PATH):
        print(f"\n[INFO] {DATA_PATH} not found - Generating synthetic data ...")
        df = generate_synthetic_data()
        df.to_csv(DATA_PATH, index=False)
        print(f"[INFO] Synthetic data saved to {DATA_PATH}")
    else:
        print(f"[INFO] Data file found at {DATA_PATH}")

def load_data(file_path=DATA_PATH):
    """
    Load credit data from a CSV file.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def basic_eda(df):
    """
    Print comprehensive exploratory data analysis info to console only.
    No tables are saved to disk; plots are saved in detailed_eda.
    """
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS (Console Summary)")
    print("="*60)

    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nNumerical Summary:")
    print(df.describe(include='all'))

    print("\nMissing Values (non-zero):")
    missing = df.isnull().sum()
    missing_df = missing[missing > 0]
    if len(missing_df) == 0:
        print("No missing values found.")
    else:
        print(missing_df)

    print("\nTarget distribution:")
    target_counts = df['default'].value_counts()
    target_pct = df['default'].value_counts(normalize=True) * 100
    print(pd.DataFrame({'Count': target_counts, 'Percentage': target_pct}))

    if 0 in target_counts and 1 in target_counts and target_counts[0] > 0:
        imbalance_ratio = target_counts[1] / target_counts[0]
    else:
        imbalance_ratio = float('nan')
    print(f"\nClass Imbalance Ratio (Default/Non-Default): {imbalance_ratio:.3f}")

def detailed_eda(df):
    """
    Perform detailed EDA and save plots to results/ only (no tables).
    Saves:
      - results/eda_12panel_overview.png
      - results/eda_correlation_heatmap.png
    """
    print("\n" + "="*60)
    print("DETAILED EDA WITH VISUALIZATIONS (Saving to results/)")
    print("="*60)

    plt.style.use('default')
    sns.set_palette("husl")

    # Combined 12-panel figure
    fig = plt.figure(figsize=(20, 15))

    # 1. Target distribution (pie chart)
    plt.subplot(3, 4, 1)
    target_counts = df['default'].value_counts()
    plt.pie(target_counts.values, labels=['Non-Default', 'Default'], autopct='%1.1f%%', startangle=90)
    plt.title('Target Distribution')

    # 2. Age distribution
    plt.subplot(3, 4, 2)
    plt.hist(df['age'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    # 3. Income distribution
    plt.subplot(3, 4, 3)
    plt.hist(df['income'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Frequency')

    # 4. Loan amount distribution
    plt.subplot(3, 4, 4)
    plt.hist(df['loan_amount'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Loan Amount Distribution')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')

    # 5. Credit utilization distribution
    plt.subplot(3, 4, 5)
    plt.hist(df['credit_utilization'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Credit Utilization Distribution')
    plt.xlabel('Credit Utilization')
    plt.ylabel('Frequency')

    # 6. Employment years distribution
    plt.subplot(3, 4, 6)
    plt.hist(df['employment_years'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Employment Years Distribution')
    plt.xlabel('Employment Years')
    plt.ylabel('Frequency')

    # 7. Age vs Default (box plot)
    plt.subplot(3, 4, 7)
    df.boxplot(column='age', by='default', ax=plt.gca())
    plt.title('Age by Default Status')
    plt.suptitle('')

    # 8. Income vs Default (box plot)
    plt.subplot(3, 4, 8)
    df.boxplot(column='income', by='default', ax=plt.gca())
    plt.title('Income by Default Status')
    plt.suptitle('')

    # 9. Credit utilization vs Default (box plot)
    plt.subplot(3, 4, 9)
    df.boxplot(column='credit_utilization', by='default', ax=plt.gca())
    plt.title('Credit Utilization by Default Status')
    plt.suptitle('')

    # 10. Loan amount vs Default (box plot)
    plt.subplot(3, 4, 10)
    df.boxplot(column='loan_amount', by='default', ax=plt.gca())
    plt.title('Loan Amount by Default Status')
    plt.suptitle('')

    # 11. Correlation heatmap (in panel)
    plt.subplot(3, 4, 11)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix')

    # 12. Default rate by employment years
    plt.subplot(3, 4, 12)
    employment_bins = pd.cut(df['employment_years'], bins=5)
    default_by_employment = df.groupby(employment_bins)['default'].mean()
    default_by_employment.plot(kind='bar', ax=plt.gca())
    plt.title('Default Rate by Employment Years')
    plt.xlabel('Employment Years (binned)')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'eda_12panel_overview.png')
    plt.savefig(fig_path)
    plt.close(fig)

    # Separate correlation heatmap (standalone)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix')
    corr_path = os.path.join(RESULTS_DIR, 'eda_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()

def feature_analysis(df):
    """
    Print per-feature stats by target to console only (no file outputs).
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS (Console Only)")
    print("="*60)

    feature_cols = [col for col in df.columns if col != 'default']
    for feature in feature_cols:
        stats = df.groupby('default')[feature].agg(['count', 'mean', 'std', 'min', 'max'])
        print(f"\n{feature.upper()}:\n{stats}")

def outlier_analysis(df):
    """
    Identify potential outliers and print summary to console only (no file outputs).
    """
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS (Console Only)")
    print("="*60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'default':
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outlier_count / len(df)) * 100
        print(f"{col}: outliers={outlier_count} ({outlier_pct:.2f}%), bounds=[{lower_bound:.2f}, {upper_bound:.2f}]")

def comprehensive_eda(df):
    """
    Run all EDA functions.
    - Console: summaries, feature stats, outlier info
    - Files in results/: plots only (no tables)
    """
    basic_eda(df)
    detailed_eda(df)
    feature_analysis(df)
    outlier_analysis(df)

    print("\n" + "="*60)
    print("EDA COMPLETE - plots saved to results/")
    print("="*60)

if __name__ == "__main__":
    ensure_data_exists()
    df = load_data()
    comprehensive_eda(df)