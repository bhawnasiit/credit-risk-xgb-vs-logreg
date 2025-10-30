# Credit Risk Model: XGBoost vs Logistic Regression

A comprehensive machine learning project comparing XGBoost and Logistic Regression for credit risk assessment using synthetic credit data.

## 🚀 Project Overview

This repository demonstrates the complete workflow of building and comparing credit risk models using two popular algorithms:
- **XGBoost** (Gradient Boosting)
- **Logistic Regression**

The project includes data generation, exploratory data analysis, feature engineering, model training, evaluation, and comparison.

## 📁 Project Structure

```
credit-risk-xgb-vs-logreg/
├── data/                       # Data directory
│   └── credit_data.csv         # Synthetic credit data (auto-generated)
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── src/                        # Source code modules
│   ├── data_prep.py           # Data generation, loading & EDA
│   ├── feature_engineering.py # Feature engineering functions
│   ├── model_training.py      # Model training and evaluation
│   └── utils.py               # Utility functions
├── models/                     # Saved model files
│   ├── xgboost_model.pkl
│   └── logistic_regression_model.pkl
├── results/                    # Results and outputs
│   ├── model_comparison.html
│   └── feature_importance.png
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhawnasiit/credit-risk-xgb-vs-logreg.git
   cd credit-risk-xgb-vs-logreg
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Generate and explore data:**
   ```bash
   python src/data_prep.py
   ```

2. **Run the complete analysis:**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

## 📊 Features

- **Synthetic Data Generation**: Creates realistic credit data for demonstration
- **Comprehensive EDA**: 12+ visualizations and statistical analysis
- **Feature Engineering**: Advanced feature creation and selection
- **Model Comparison**: Side-by-side comparison of XGBoost vs Logistic Regression
- **Performance Metrics**: ROC curves, confusion matrices, feature importance
- **Reproducible Results**: Fixed random seeds for consistent results

## 📈 Key Insights

- Class imbalance handling techniques
- Feature importance analysis
- Model interpretability comparison
- Performance trade-offs between algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@bhawnasiit](https://github.com/yourusername)

## 🙏 Acknowledgments

- scikit-learn team for the excellent ML library
- XGBoost developers for the powerful gradient boosting framework
- The data science community for inspiration and best practices
```

## 3. Update .gitignore

```gitignore:.gitignore
# Data files
data/*.csv
data/*.json
data/*.parquet

# Model files
models/*.pkl
models/*.joblib
models/*.h5

# Results and outputs
results/*.html
results/*.png
results/*.jpg
results/*.pdf

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
```

## 4. Create Additional Folders

You'll need to create these empty folders in your project:

```bash
mkdir notebooks
mkdir models
mkdir results
```

## 5. Future File Structure Plan

As we build more components, here's where everything will go:

### 📁 **notebooks/** (Jupyter notebooks for exploration)
- `01_data_exploration.ipynb` - EDA and data understanding
- `02_feature_engineering.ipynb` - Feature creation and selection
- `03_model_comparison.ipynb` - Model training and comparison
- `04_model_interpretation.ipynb` - SHAP values, feature importance

### 📁 **src/** (Source code modules)
- `data_prep.py` ✅ (Already done)
- `feature_engineering.py` - Feature creation functions
- `model_training.py` - Model training and evaluation
- `model_comparison.py` - Model comparison utilities
- `utils.py` - Helper functions

### 📁 **models/** (Saved models)
- `xgboost_model.pkl`
- `logistic_regression_model.pkl`
- `feature_scaler.pkl`

### 📁 **results/** (Outputs and visualizations)
- `model_comparison.html`
- `roc_curves.png`
- `feature_importance.png`
- `confusion_matrices.png`

## 6. Next Steps

When we add new components, I'll remind you to:

1. **Update the appropriate folder** (notebooks/, src/, models/, results/)
2. **Update requirements.txt** if new dependencies are added
3. **Update README.md** if new features are added
4. **Update .gitignore** if new file types need to be ignored

## 7. GitHub Repository Setup

1. Create a new repository on GitHub
2. Initialize locally:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Credit risk model project setup"
   git branch -M main
   git remote add origin https://github.com/yourusername/credit-risk-xgb-vs-logreg.git
   git push -u origin main
   ```

Would you like me to help you create any of the additional files (like the Jupyter notebooks or the next Python modules) as we continue building the project?
