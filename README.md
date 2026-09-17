# 🏦 Bank Loan Default Prediction

An explainable machine learning project that predicts whether a bank loan applicant is likely to **default or repay** using **Logistic Regression**.

The project uses a realistic **synthetic dataset** containing financial and credit-risk features and demonstrates the complete machine learning workflow—from data generation and preprocessing to model training, evaluation, feature interpretation, and individual applicant prediction.

> **Note:** This project is designed for educational and case-study purposes. The dataset is synthetic and should not be used for real-world lending decisions.

---

## 📌 Project Overview

Loan default prediction is a common classification problem in credit-risk analytics.

This project builds a simple and interpretable **Logistic Regression classification model** to identify applicants who may be at higher risk of default.

The notebook demonstrates:

* Synthetic loan dataset creation
* Exploratory data analysis
* Categorical variable encoding
* Train-test splitting
* Feature scaling
* Logistic Regression model training
* Accuracy and classification-report evaluation
* Confusion matrix analysis
* Model coefficient interpretation
* Individual applicant default-risk prediction

The notebook intentionally uses Logistic Regression because its feature coefficients make the model relatively easy to interpret.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Build a binary classification model for loan default prediction.
2. Understand how applicant financial characteristics relate to default risk.
3. Prepare categorical and numerical data for machine learning.
4. Train and evaluate a Logistic Regression model.
5. Interpret model coefficients.
6. Demonstrate predictions for individual loan applicants.

---

## 📊 Dataset

The notebook generates a synthetic dataset containing **3,000 loan applicants**.

### Features

| Feature             | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `annual_income`     | Applicant's annual income, in thousands                          |
| `credit_score`      | Credit bureau score                                              |
| `loan_amount`       | Requested loan amount, in thousands                              |
| `debt_to_income`    | Existing monthly debt payments as a percentage of monthly income |
| `employment_years`  | Years at the applicant's current job                             |
| `late_payments_2yr` | Number of late payments during the previous 2 years              |
| `loan_purpose`      | Loan purpose: Home, Auto, or Personal                            |
| `default`           | Target variable: `0` = repaid, `1` = defaulted                   |

These variables are explicitly defined in the notebook as the inputs to the synthetic loan-risk case study.

---

## 🤖 Machine Learning Approach

### Algorithm

**Logistic Regression**

Logistic Regression is used to classify applicants into two categories:

* `0` → Repaid
* `1` → Defaulted

## The model learns a coefficient for each feature. Positive coefficients push predictions toward default, while negative coefficients push predictions toward repayment.

## 🔄 Machine Learning Workflow

```text
Synthetic Data Generation
          ↓
Exploratory Data Analysis
          ↓
Categorical Encoding
          ↓
Train / Test Split
          ↓
Feature Scaling
          ↓
Logistic Regression
          ↓
Model Predictions
          ↓
Model Evaluation
          ↓
Feature Interpretation
          ↓
Individual Applicant Prediction
```

---

## 🛠️ Technologies Used

* **Python 3.10**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

The notebook imports `numpy`, `pandas`, `matplotlib`, and Scikit-learn components for splitting, scaling, Logistic Regression, and evaluation.

---

## ⚙️ Data Preprocessing

Before training the model, the notebook performs three main preprocessing steps:

### 1. One-Hot Encoding

The categorical `loan_purpose` variable is converted into numerical features using one-hot encoding.

```python
df_model = pd.get_dummies(
    df,
    columns=['loan_purpose'],
    drop_first=True
)
```

### 2. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A stratified split is used to preserve the target-class distribution.

### 3. Feature Scaling

`StandardScaler` is applied to the numerical model inputs before training the Logistic Regression model.

---

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the proportion of correctly classified applicants.

### Classification Report

The notebook generates classification metrics for:

* Repaid
* Defaulted

### Confusion Matrix

The confusion matrix shows:

* Actual repaid vs. predicted repaid
* Actual repaid vs. predicted default
* Actual defaulted vs. predicted repaid
* Actual defaulted vs. predicted default

These evaluation steps are implemented directly in the notebook.

---

## 🔍 Feature Importance / Model Interpretation

Instead of using a separate feature-importance algorithm, this project examines the **Logistic Regression coefficients**.

The notebook sorts the learned coefficients by their absolute magnitude and visualizes them.

The case study describes the main patterns as:

* Lower credit scores → higher default risk
* More late payments → higher default risk
* Higher income → lower default risk
* Longer employment tenure → lower default risk
* Lower debt-to-income → lower default risk

## These relationships are specific to the synthetic data-generation rules and model used in this educational notebook.

## 🧪 Live Prediction

The notebook includes a reusable function:

```python
predict_default(
    annual_income,
    credit_score,
    loan_amount,
    debt_to_income,
    employment_years,
    late_payments_2yr,
    loan_purpose='Auto'
)
```

The function:

1. Creates a data row for an applicant.
2. Encodes the loan purpose.
3. Applies the trained scaler.
4. Calculates the predicted default probability.
5. Produces a classification based on the model's probability threshold.

Example:

```python
predict_default(
    annual_income=95,
    credit_score=780,
    loan_amount=15,
    debt_to_income=18,
    employment_years=10,
    late_payments_2yr=0,
    loan_purpose='Home'
)
```

---

## 📁 Repository Structure

```text
bank-loan-default-prediction/
│
├── bank_loan_default_logistic_regression.ipynb
├── README.md
└── LICENSE
```

---

## 🚀 How to Run

### Option 1 — Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Run the cells from top to bottom.
3. Review the generated dataset and visualizations.
4. Train the Logistic Regression model.
5. Evaluate the model.
6. Modify the applicant information in the prediction section to test different scenarios.

### Option 2 — Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/<your-username>/bank-loan-default-prediction.git
cd bank-loan-default-prediction
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
bank_loan_default_logistic_regression.ipynb
```

---

## 💡 Key Learning Outcomes

After completing this project, you should understand:

* How binary classification works
* How Logistic Regression can be applied to credit-risk problems
* How categorical variables are encoded
* Why feature scaling is useful
* How to split data into training and testing sets
* How to evaluate classification models
* How to interpret Logistic Regression coefficients
* How to generate predictions for individual applicants

---

## ⚠️ Limitations

This project uses **synthetic data generated specifically for the case study** rather than actual bank loan records.

Therefore:

* Model performance should not be interpreted as real-world bank performance.
* The synthetic relationships may not represent actual borrower behavior.
* The model is intended for learning and demonstration.
* Real lending systems require appropriate validation, governance, privacy controls, fairness assessment, and regulatory compliance.

The notebook itself identifies the need for bias testing and real applicant/repayment data before real-world deployment.

---

## 🔮 Future Improvements

Possible extensions include:

* Test additional classification algorithms
* Add ROC-AUC and precision-recall analysis
* Perform cross-validation
* Tune model hyperparameters
* Handle class imbalance
* Add probability calibration
* Compare Logistic Regression with tree-based models
* Add SHAP-based model explanations
* Build a Streamlit prediction interface
* Use a properly governed real-world dataset for further research

---

## 👤 Author

**Hirday Kohli**

GitHub: `https://github.com/<your-username>`

---

## 📄 License

This project can be distributed under the **MIT License**. Add a `LICENSE` file to the repository if you choose to publish it under MIT.

---

⭐ If you find this project useful for learning machine learning and credit-risk modeling, consider starring the repository.
