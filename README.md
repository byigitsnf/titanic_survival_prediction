# Titanic Survival Prediction – KDD Project

A machine learning project implementing the **Knowledge Discovery in Databases (KDD)** process on the Titanic passenger dataset. The project focuses on data preprocessing, exploratory data analysis, feature engineering, classification, and model evaluation to predict passenger survival.

## Project Overview

The objective of this project is to analyze passenger information and build machine learning models that predict whether a passenger survived the Titanic disaster.

The project follows a structured KDD workflow, covering the transformation of raw data into meaningful insights and predictive models.

**Problem Type:** Binary Classification  
**Target Variable:** `survived`  
**Dataset:** Titanic Dataset (OpenML ID: 40945, 1,309 passenger records)

## Technologies & Libraries

- **Python** – Main programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical data visualization
- **Scikit-learn** – Data preprocessing, machine learning, and model evaluation

## Project Workflow

### 1. Data Collection and Understanding

- Loaded the Titanic dataset from a local CSV file or OpenML.
- Examined dataset dimensions, data types, descriptive statistics, and missing values.
- Analyzed the distribution of the target variable.

### 2. Data Preprocessing and Cleaning

- Identified and handled missing values.
- Applied KNN Imputation (`k=5`) to estimate missing age values.
- Used median and mode-based imputation for selected features.
- Detected outliers using IQR and Z-score methods.
- Applied log transformation and capping to reduce the impact of extreme values.
- Removed irrelevant features and potential data leakage variables.

### 3. Feature Engineering

Created and transformed features to improve the predictive capability of the models:

- **Title Extraction:** Extracted passenger titles from names and grouped rare titles.
- **Family Size:** Combined `sibsp` and `parch` into a single `family_size` feature.
- **Cabin Information:** Created a binary `has_cabin` feature.
- **Categorical Encoding:** Applied One-Hot Encoding and binary encoding where appropriate.
- **Feature Scaling:** Used StandardScaler for numerical feature standardization.

### 4. Exploratory Data Analysis (EDA)

Performed statistical analysis and visual exploration to investigate relationships between passenger characteristics and survival outcomes.

Visualizations include:

- Missing value distribution
- Age distribution and survival patterns
- Correlation heatmap
- Survival rates by gender, passenger class, and title
- Outlier analysis using boxplots
- Age versus ticket fare distribution
- Family size and survival analysis

### 5. Model Development

Three classification algorithms were implemented and compared:

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model |
| Decision Tree | Rule-based classification model |
| Random Forest | Ensemble learning algorithm |

Hyperparameter optimization was performed using **GridSearchCV** with 5-fold Stratified Cross-Validation.

The models were trained using an 80/20 stratified train-test split.

### 6. Model Evaluation

Model performance was evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Additional evaluation techniques include:

- Confusion matrices
- ROC curves
- Cross-validation performance comparison
- Random Forest feature importance
- Logistic Regression coefficient analysis

## Key Findings

The analysis highlights the importance of passenger characteristics in survival prediction.

The main observations include:

- Gender, passenger title, and passenger class are among the strongest survival-related features.
- Feature engineering helps extract meaningful information from raw passenger data.
- Comparing multiple classification algorithms provides a better understanding of model performance.
- Random Forest and Logistic Regression achieved a similar overall performance balance in the implemented experiments.

## Repository Structure

```text
titanic_kdd/
│
├── figs/          # Generated data analysis and model evaluation plots
├── titanic.csv    # Titanic dataset
├── titanic.py     # Main KDD and machine learning implementation
└── README.md      # Project documentation
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/byigitsnf/titanic_kdd.git
cd titanic_kdd
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the Project

```bash
python titanic.py
```

The application first attempts to load `titanic.csv` from the project directory. If the file is unavailable, it retrieves the dataset from OpenML, requiring an internet connection.

Generated visualizations are saved automatically in the `figs/` directory.

## Project Objectives

This project was developed to gain practical experience in:

- Applying the KDD methodology to a real-world dataset.
- Data cleaning and preprocessing techniques.
- Feature engineering and exploratory data analysis.
- Implementing and comparing classification algorithms.
- Hyperparameter optimization and cross-validation.
- Evaluating machine learning models using multiple performance metrics.
- Interpreting model results and identifying important features.

## Author

**byigitsnf**

[GitHub Profile](https://github.com/byigitsnf)

[Project Repository](https://github.com/byigitsnf/titanic_kdd)
