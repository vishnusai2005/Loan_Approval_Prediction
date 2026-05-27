# Loan_Approval_Prediction[README.md](https://github.com/user-attachments/files/28310185/README.md)
# 🏦 Loan Approval Prediction

A complete end-to-end Machine Learning project that predicts whether a loan application will be approved or rejected, using classical ML algorithms with full EDA, feature engineering, and hyperparameter tuning.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Techniques Used In Depth](#techniques-used-in-depth)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Missing Value Imputation](#2-missing-value-imputation)
  - [3. Outlier Treatment via Log Transformation](#3-outlier-treatment-via-log-transformation)
  - [4. Categorical Feature Encoding](#4-categorical-feature-encoding)
  - [5. Feature Engineering](#5-feature-engineering)
  - [6. Train-Test Split](#6-train-test-split)
  - [7. K-Nearest Neighbors (KNN)](#7-k-nearest-neighbors-knn)
  - [8. Support Vector Machine (SVM)](#8-support-vector-machine-svm)
  - [9. Decision Tree Classifier](#9-decision-tree-classifier)
  - [10. Model Evaluation Metrics](#10-model-evaluation-metrics)
  - [11. Leave-One-Feature-Out (LOFO) Feature Importance](#11-leave-one-feature-out-lofo-feature-importance)
  - [12. Hyperparameter Tuning — GridSearchCV](#12-hyperparameter-tuning--gridsearchcv)
  - [13. Hyperparameter Tuning — RandomizedSearchCV](#13-hyperparameter-tuning--randomizedsearchcv)
  - [14. Correlation Heatmap](#14-correlation-heatmap)
- [Model Comparison](#model-comparison)
- [Results & Output Files](#results--output-files)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Project Overview

This project solves a **binary classification problem**: given applicant details (income, credit history, loan amount, property area, etc.), predict whether the bank should **approve (Y)** or **reject (N)** the loan.

**Key Goals:**
- Clean and preprocess real-world tabular data with missing values and skewed distributions
- Engineer meaningful features from raw financial data
- Build and compare three classifiers: KNN, SVM, Decision Tree
- Tune hyperparameters to squeeze out the best possible accuracy
- Identify which features drive loan approval decisions the most

---

## Dataset

| Split | File | Rows | Columns |
|-------|------|------|---------|
| Training | `loan_sanction_train.csv` | 614 | 13 |
| Testing | `loan_sanction_test.csv` | 367 | 12 |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | Categorical | Unique application identifier |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Categorical | 0, 1, 2, 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numerical | Monthly income of primary applicant |
| `CoapplicantIncome` | Numerical | Monthly income of co-applicant |
| `LoanAmount` | Numerical | Requested loan in thousands |
| `Loan_Amount_Term` | Numerical | Repayment period in months |
| `Credit_History` | Binary | 1 = good history, 0 = bad history |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Loan_Status` | **Target** | Y (approved) / N (rejected) |

---

## Project Pipeline

```
Raw CSV Data
     │
     ▼
Exploratory Data Analysis (EDA)
     │
     ▼
Missing Value Imputation (Mode / Median)
     │
     ▼
Outlier Treatment (Log / Sqrt Transformation)
     │
     ▼
Feature Encoding (Binary Map + OHE + Label Encoding)
     │
     ▼
Feature Engineering (Total Income, Log features)
     │
     ▼
Train-Test Split (80/20)
     │
     ├─► KNN  ──────────────┐
     ├─► SVM  ──────────────┤──► Model Evaluation + Comparison
     └─► Decision Tree ─────┘
                            │
                            ▼
                   LOFO Feature Importance
                            │
                            ▼
               Hyperparameter Tuning (Grid + Random)
                            │
                            ▼
                  Final Predictions on Test Set
```

---

## Techniques Used In Depth

---

### 1. Exploratory Data Analysis (EDA)

EDA is the first and most critical phase of any ML project. Before building models, you need to **understand the data**: its distribution, patterns, correlations, and class imbalances.

**What was done in this notebook:**

**Univariate Analysis** — studied each feature individually:
- `value_counts(normalize=True)` on categorical columns like `Loan_Status`, `Gender`, `Married`, `Credit_History` to see class proportions as bar charts
- `sns.distplot()` and `.plot.box()` on numerical columns like `ApplicantIncome` and `CoapplicantIncome` to see the distribution shape and spot outliers

**Bivariate Analysis** — studied each feature against the target `Loan_Status`:
- `pd.crosstab()` between categorical columns and `Loan_Status`, then normalized and plotted as stacked bar charts. This revealed that:
  - Applicants with a **good credit history** had much higher approval rates
  - **Married** applicants were more likely to get approval
  - **Semiurban** properties had a higher loan approval ratio
- Income binning (`pd.cut()`) was used to group continuous incomes into `Low`, `Average`, `High`, `Very High` buckets and then plot approval rates per bin

**Why EDA matters:** It shapes every downstream decision — what features to keep, how to encode them, whether transformations are needed, and which patterns to leverage in modeling.

---

### 2. Missing Value Imputation

Real datasets are rarely complete. This dataset had missing values in 7 out of 12 features. Imputation fills those gaps without dropping valuable rows.

**Missing counts in training data:**

| Feature | Missing Count | Strategy Used |
|---------|--------------|---------------|
| `Gender` | 13 | Mode |
| `Married` | 3 | Mode |
| `Dependents` | 15 | Mode |
| `Self_Employed` | 32 | Mode |
| `Credit_History` | 50 | Mode |
| `Loan_Amount_Term` | 14 | Mode |
| `LoanAmount` | 22 | Median |

**Why Mode for Categoricals?**
Mode is the most frequently occurring value. For binary/categorical columns like `Gender` (mostly Male), imputing with mode maintains the dominant distribution rather than introducing artificial noise.

**Why Median for `LoanAmount`?**
`LoanAmount` is a skewed numerical column (right-skewed, as shown in EDA). The median is **resistant to outliers** — it doesn't get dragged by a few very high loan values the way the mean would. This makes it a more representative central value for imputation.

**Test set imputation:** Values were imputed using **train set statistics** (not test set), which is important to prevent data leakage — the test set must remain completely unseen information.

---

### 3. Outlier Treatment via Log Transformation

Skewed distributions cause problems for many ML algorithms. Outliers in income and loan amount can dominate gradient computations, distance calculations (in KNN), and decision boundaries (in SVM).

**Transformations applied:**

| Feature | Transformation | Reason |
|---------|---------------|--------|
| `LoanAmount` | `np.log(LoanAmount)` → `LoanAmount_log` | Right-skewed; log compresses large values |
| `ApplicantIncome` | `np.log(ApplicantIncome)` → `ApplicantIncome_log` | Heavy right tail due to high earners |
| `CoapplicantIncome` | `np.sqrt(CoapplicantIncome)` → `CoapplicantIncome_log` | Contains many zeros; sqrt handles zero safely, log does not |
| `CoapplicantIncome` | `np.log(x + 1)` → `CoapplicantIncome_log_offset` | Log(0) is undefined; adding +1 offset makes it safe |

After comparison histograms were plotted (original vs. transformed), the sqrt transform was retained for CoapplicantIncome and the original columns were dropped.

**Why this matters:** After transformation, the distribution becomes more Gaussian (bell-curve shaped), which helps algorithms that assume normality and makes distance-based models (KNN) and margin-based models (SVM) perform more reliably.

---

### 4. Categorical Feature Encoding

Machine learning models require **numerical inputs**. Categorical text values must be converted. Three encoding strategies were used:

#### Binary Encoding (Manual Mapping)
For binary/ordinal columns, direct integer mapping was applied:

```python
train['Gender']      = train['Gender'].map({'Male': 1, 'Female': 0})
train['Married']     = train['Married'].map({'Yes': 1, 'No': 0})
train['Education']   = train['Education'].map({'Graduate': 1, 'Not Graduate': 0})
train['Self_Employed'] = train['Self_Employed'].map({'Yes': 1, 'No': 0})
```

This is preferred over Label Encoding for truly binary features since it is semantically clear and doesn't introduce false ordinal relationships.

#### One-Hot Encoding (OHE)
Applied to `Property_Area` (Urban / Semiurban / Rural) using `pd.get_dummies()` with `drop_first=True`:

```python
train_df_encoded = pd.get_dummies(train, columns=['Property_Area'], drop_first=True)
```

OHE creates one binary column per category. `drop_first=True` drops one column to avoid the **dummy variable trap** (multicollinearity — where one column is perfectly predictable from the others).

**Result:** `Property_Area_Semiurban` and `Property_Area_Urban` columns were created; Rural became the implicit baseline (both = 0).

#### Label Encoding
Applied using `sklearn.preprocessing.LabelEncoder`:

```python
le = LabelEncoder()
train['Property_Area'] = le.fit_transform(train['Property_Area'])
```

Label Encoding assigns integer codes (0, 1, 2) to categories. It was used here on the Label-Encoded version of the dataset (used as input to the first round of models), while OHE was used on the encoded DataFrame version. **Note:** Label Encoding implies an ordinal relationship (0 < 1 < 2) which may not be appropriate for Property_Area — OHE is generally safer for nominal categories.

---

### 5. Feature Engineering

Raw features were combined or transformed to create more informative signals:

**Total Income:**
```python
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
```
A bank evaluates the household's combined repayment capacity — not just the primary applicant. This feature captures joint financial strength.

**Income and Loan Binning:**
`pd.cut()` was used to convert continuous income/loan amount into discrete buckets (Low, Average, High, Very High). These bins were used **only during EDA** for visualization and were dropped before modeling to avoid ordinal encoding assumptions.

---

### 6. Train-Test Split

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

The labeled training set (614 rows) was split:
- **80% training** (≈491 rows) — used to fit the model
- **20% validation/test** (≈123 rows) — used to evaluate unseen performance

`random_state=42` ensures reproducibility — the same split is generated on every run, making results comparable.

**Why not use the actual test set for evaluation?** The test CSV (`loan_sanction_test.csv`) has no `Loan_Status` labels — it is a blind prediction set. All model comparisons are done on the 20% holdout split from the training data.

---

### 7. K-Nearest Neighbors (KNN)

**Concept:** KNN is a lazy, non-parametric algorithm. It makes predictions by finding the **K nearest training points** (measured by Euclidean distance) to a query point and taking a majority vote of their labels.

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
```

**K-value Tuning:**
A loop was run over K = 1 to 39:

```python
scores = []
for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
plt.plot(range(1, 40), scores)
```

The accuracy curve was plotted to identify the optimal K. K=25 was chosen as the best value — small K values overfit (memorize training data), while very large K values underfit (too much smoothing).

**Key characteristics:**
- No explicit training step (model just stores data)
- Sensitive to feature scale — log-transformations help equalize feature ranges
- Computationally expensive at prediction time for large datasets

---

### 8. Support Vector Machine (SVM)

**Concept:** SVM finds the **optimal hyperplane** that separates classes with the **maximum margin** — the widest possible gap between the decision boundary and the nearest data points from each class (called support vectors).

```python
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
```

**Linear Kernel:** Projects data in original feature space and finds a straight decision boundary. Effective when data is approximately linearly separable.

**Hyperparameter tuning of SVM (GridSearchCV):**

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}
```

- **C (Regularization):** Controls the trade-off between maximizing the margin and minimizing misclassifications. Low C = wider margin (more tolerant of misclassification). High C = narrower margin (tries harder to classify all points correctly, risking overfitting).
- **Kernel:** `linear` works in original space; `rbf` (Radial Basis Function) projects to a higher-dimensional space to handle non-linearly separable data; `poly` uses polynomial projections.
- **Gamma:** Controls the influence radius of a single training example. High gamma = local influence; low gamma = global influence.

---

### 9. Decision Tree Classifier

**Concept:** A Decision Tree splits data recursively at the feature and threshold that best separates the classes, building a tree of if-else rules until stopping criteria are met.

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt.fit(x_train, y_train)
```

**Entropy (Information Gain) Criterion:**
At each split, the algorithm chooses the feature that reduces **entropy** the most:

```
Entropy(S) = -Σ p_i * log2(p_i)
```

A pure node (all one class) has entropy = 0. A maximally mixed node (50/50 split) has entropy = 1. The feature that produces the highest **Information Gain** (entropy before split minus weighted entropy after) is chosen.

**Hyperparameters tuned:**
- `max_depth` — limits tree depth to prevent overfitting
- `min_samples_split` — minimum samples required to split an internal node
- `min_samples_leaf` — minimum samples required at a leaf node
- `max_leaf_nodes` — maximum number of terminal nodes

**Tree visualization:**
```python
from sklearn.tree import plot_tree
plot_tree(dt, feature_names=x.columns, class_names=['0', '1'], filled=True, rounded=True)
```
This renders the full decision path — you can trace exactly why a loan was approved or rejected.

---

### 10. Model Evaluation Metrics

After each model's predictions, a full classification report was generated:

```python
from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_test, y_pred))
```

**Metrics explained:**

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Accuracy** | (TP + TN) / Total | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of all predicted approvals, how many were actually approved |
| **Recall** | TP / (TP + FN) | Of all actual approvals, how many did the model catch |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall |
| **Support** | — | Number of actual samples per class in test set |

**Why F1 matters here:** The dataset is imbalanced (more approvals than rejections). Accuracy alone can be misleading — a model that always predicts "Approved" would look ~69% accurate. F1-score penalizes models that ignore the minority class.

---

### 11. Leave-One-Feature-Out (LOFO) Feature Importance

LOFO is a model-agnostic technique for measuring how much each feature contributes to model accuracy.

**Algorithm:**
1. Train the model on all features → record baseline accuracy
2. For each feature, **remove it**, retrain, and record new accuracy
3. Feature importance = `baseline_accuracy - reduced_accuracy`

```python
feature_importance = {}
for feature in x.columns:
    x_train_reduced = x_train.drop(columns=[feature])
    x_test_reduced  = x_test.drop(columns=[feature])
    dt.fit(x_train_reduced, y_train)
    reduced_accuracy = accuracy_score(y_test, dt.predict(x_test_reduced))
    feature_importance[feature] = dt_accuracy - reduced_accuracy

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
```

**Interpretation:**
- A large positive drop → feature was critical
- Near-zero or negative drop → feature was adding noise or was redundant

This was applied to **both Decision Tree and SVM** independently — and the top features (`Credit_History`, `ApplicantIncome_log`, `LoanAmount_log`, `Loan_Amount_Term`, `CoapplicantIncome_log`) were selected as the final feature set for hyperparameter-tuned models.

**Why LOFO over built-in feature importances?**
Built-in `.feature_importances_` in Decision Trees can be biased toward high-cardinality features. LOFO uses actual held-out accuracy as its measure, making it more reliable and model-agnostic.

---

### 12. Hyperparameter Tuning — GridSearchCV

Grid Search performs an **exhaustive search** over a manually defined grid of hyperparameter values, evaluating every combination using cross-validation.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_leaf_nodes': [10, 20, 50]
}

grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
```

**Key parameters:**
- `cv=5` — 5-fold cross-validation. Training data is split into 5 folds; each fold is used as validation once, giving a more robust accuracy estimate
- `scoring='accuracy'` — metric used to rank parameter combinations
- `n_jobs=-1` — uses all available CPU cores for parallel computation

**Limitation:** Grid Search evaluates `5 × 4 × 4 × 3 = 240` combinations × 5 folds = **1,200 model fits**. For larger grids, this becomes computationally expensive.

---

### 13. Hyperparameter Tuning — RandomizedSearchCV

Random Search samples a fixed number of parameter settings from specified distributions, rather than exhaustively trying all combinations.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_leaf_nodes': randint(10, 100)
}

random_search = RandomizedSearchCV(
    estimator=dt_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
random_search.fit(x_train, y_train)
```

`n_iter=100` means only 100 random combinations are tried (vs. 240+ for Grid Search).

**When is Random Search better?**
Research (Bergstra & Bengio, 2012) shows that when only a few hyperparameters matter, random search finds good solutions faster because it explores a wider range of values for every parameter. Grid Search wastes time testing exhaustive combinations of unimportant parameters.

**scipy.stats.randint:** Provides a discrete uniform distribution for integer hyperparameters. This is more flexible than a fixed list — it samples from a continuous range.

---

### 14. Correlation Heatmap

```python
matrix = numerical_df.corr()
sns.heatmap(matrix, vmax=0.8, square=True, cmap='coolwarm')
```

A correlation matrix shows **pairwise linear correlations** between all numerical features, ranging from -1 (perfect inverse) to +1 (perfect direct correlation).

**Purpose here:** Identify multicollinearity — if two features are highly correlated (e.g., `ApplicantIncome` and `LoanAmount`), they carry redundant information and one can be dropped without loss. The heatmap also reveals which features correlate with `Loan_Status`, giving a preview of feature importance.

---

## Model Comparison

| Model | Approach | Strengths in this project |
|-------|----------|--------------------------|
| **KNN (k=25)** | Distance-based voting | Simple baseline; good with normalized features |
| **SVM (linear kernel)** | Max-margin hyperplane | Robust to high dimensions; good with log-transformed features |
| **Decision Tree (entropy, depth=3)** | Recursive binary splits | Interpretable; tree visualization shows exact rules |

All three models were compared by plotting accuracy using a bar chart:

```python
models = ['KNN', 'SVM', 'Decision Tree']
accuracy = [knn_accuracy, svm_accuracy, dt_accuracy]
plt.bar(models, accuracy)
```

After hyperparameter tuning, Decision Tree and SVM were re-evaluated on the same 5-feature subset selected via LOFO.

---

## Results & Output Files

| File | Description |
|------|-------------|
| `test_predictions_knn.csv` | KNN predictions on the 367-row test set |
| `test_predictions_svm.csv` | SVM predictions on the 367-row test set |
| `test_predictions_dt.csv` | Decision Tree predictions on the 367-row test set |

All output files contain two columns: `Loan_ID` and `Loan_Status` (predicted).

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Load_Approval_Prediction.ipynb
```

Ensure `loan_sanction_train.csv` and `loan_sanction_test.csv` are in the same directory and update the file paths in the data-loading cells accordingly.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

---

## Author

**Vydhyam Vishnusai**  
B.Tech CSE (AI & ML) — Mohan Babu University, Tirupati  
Student ID: 23102A010034  

---

