# Diabetes Risk Intelligence

## a. Problem Statement

Diabetes is one of the most prevalent chronic diseases worldwide. Early detection of diabetes or prediabetes is critical for timely intervention, lifestyle changes, and preventing severe complications such as heart disease or vision loss.

This project frames a real-world healthcare screening problem: predicting diabetes or prediabetes risk from accessible lifestyle and health survey indicators. The goal is to demonstrate how machine-learning classification models can support early risk assessment.

> **Important:** This project is an educational machine-learning demonstration and is not a medical diagnostic tool.

---

## b. Dataset Description

- **Source**: Kaggle — [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), derived from the CDC's BRFSS 2015 survey.
- **Original size**: 253,680 rows and 22 columns.
- **Features**: 21 numeric/binary features covering lifestyle choices and health indicators such as smoking, physical activity, BMI, high blood pressure, and cholesterol.
- **Target variable**: `Diabetes_binary`
  - `0` = no diabetes
  - `1` = prediabetes or diabetes
- **Class imbalance**: The dataset is heavily imbalanced, with approximately 86% of observations belonging to the negative class. Therefore, accuracy alone can be misleading. This project considers **AUC, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)** when comparing models.

### Final Experimental Dataset

To make the training process lighter while retaining a representative class distribution, a **stratified sample of 50,000 observations** was selected from the original dataset.

The same stratified working sample was then split into:

```text
Original dataset
253,680 rows
      │
      ▼
Stratified working sample
50,000 rows
      │
      ▼
Stratified 80/20 split
      │
      ├───────────────┐
      ▼               ▼
Training set       Test set
40,000 rows        10,000 rows
      │               │
      └───────┬───────┘
              ▼
      Same split for all
          5 models
```

This ensures that the five models are compared fairly using the **same training data and the same held-out test data**.

---

## c. GitHub Repository Link

https://github.com/sqlenthusiastmaaz/diabetes-risk-intelligence

---

## d. Models Used

Five classification models were trained and evaluated using the same 40,000-row training set and the same 10,000-row held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7419 | **0.8345** | 0.3230 | **0.7782** | **0.4565** | **0.3771** |
| Random Forest | 0.7749 | 0.8204 | 0.3424 | 0.6691 | 0.4530 | 0.3587 |
| Decision Tree | 0.7184 | 0.7448 | 0.2918 | 0.7157 | 0.4146 | 0.3173 |
| Gaussian Naive Bayes | 0.7798 | 0.7933 | 0.3331 | 0.5793 | 0.4230 | 0.3164 |
| K-Nearest Neighbors | **0.8519** | 0.7264 | **0.4345** | 0.2096 | 0.2828 | 0.2288 |

### Overall Model Selection

**Logistic Regression is selected as the overall model winner.**

The selection is not based on accuracy alone. Logistic Regression achieved:

- **Highest AUC:** 0.8345
- **Highest Recall:** 0.7782
- **Highest F1 Score:** 0.4565
- **Highest MCC:** 0.3771

These metrics provide a better assessment of performance for this imbalanced classification problem.

K-Nearest Neighbors achieved the highest accuracy (**85.19%**) and highest precision (**43.45%**), but its recall was only **20.96%** and MCC was **0.2288**. This demonstrates why accuracy alone is not sufficient for evaluating this problem.

---

## e. Observations Table

| ML Model Name | Observation |
| :--- | :--- |
| **Logistic Regression** | Achieved the strongest overall performance, with the highest AUC (0.8345), Recall (77.82%), F1 Score (0.4565), and MCC (0.3771). Its combination of discriminatory ability and minority-class detection makes it the overall winner for this experiment. |
| **Decision Tree** | Achieved relatively strong Recall (71.57%), but its AUC (0.7448), F1 Score (0.4146), and MCC (0.3173) were lower than the leading models. |
| **K-Nearest Neighbors** | Achieved the highest Accuracy (85.19%) and Precision (43.45%), but very low Recall (20.96%), F1 Score (0.2828), and MCC (0.2288). This indicates that the high accuracy is largely associated with performance on the majority class. |
| **Gaussian Naive Bayes** | Achieved good Accuracy (77.98%) and moderate AUC (0.7933), but its Recall (57.93%), F1 Score (0.4230), and MCC (0.3164) were below the leading models. |
| **Random Forest** | Delivered strong overall performance, with AUC of 0.8204, Recall of 66.91%, F1 Score of 0.4530, and MCC of 0.3587. It was the second-strongest model overall and performed particularly well compared with the other non-linear models. |
| **Overall Winner** | **Logistic Regression.** It achieved the highest AUC, Recall, F1 Score, and MCC on the common 10,000-row test set, providing the strongest overall balance for this experiment. |

---

## f. Model Evaluation Metrics

Because the target variable is imbalanced, multiple evaluation metrics are used.

### Accuracy

The proportion of all predictions that are correct.

### Precision

Among observations predicted as positive, the proportion that are actually positive.

### Recall

Among actual positive observations, the proportion correctly identified by the model.

Recall is particularly important in screening-oriented applications because failing to identify a positive case can be more concerning than generating an additional false positive.

### F1 Score

The harmonic mean of Precision and Recall. It provides a single measure that balances the two.

### AUC

The Area Under the ROC Curve measures how well the model separates the two classes across classification thresholds.

### Matthews Correlation Coefficient (MCC)

MCC summarizes the quality of binary classification using all four confusion-matrix categories: true positives, true negatives, false positives, and false negatives. It is especially useful when the classes are imbalanced.

---

## g. Feature Interpretation

The Streamlit application provides feature-level analysis for supported models.

### Logistic Regression

The application displays the strongest Logistic Regression coefficients.

- A **positive coefficient** increases the model's predicted log-odds of the positive class (`Diabetes_binary = 1`).
- A **negative coefficient** decreases the model's predicted log-odds of the positive class.
- A larger absolute coefficient indicates stronger influence on the model's standardized linear decision function.

These coefficients represent **model associations and should not be interpreted as causal relationships**.

### Decision Tree and Random Forest

The application also displays feature importance for tree-based models.

Feature importance indicates how much features contribute to the model's decisions. It does not establish that a feature causes diabetes or prediabetes.

---

## h. How to Run

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/sqlenthusiastmaaz/diabetes-risk-intelligence.git
cd diabetes-risk-intelligence/ML_01/ML
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the training script**

Pre-trained model files are included in the repository, so retraining is optional.

```bash
python model/train_models.py
```

The training script generates the model files and:

```text
model/metrics_comparison.csv
```

4. **Launch the Streamlit application**

```bash
streamlit run app.py
```

The application uses the bundled `test_data.csv` and saved model/preprocessor files by default.

---

## i. Streamlit Community Cloud Deployment

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new application.
4. Select the GitHub repository and branch.
5. Set the application entry point to:

```text
ML_01/ML/app.py
```

6. Deploy.

The deployed application uses the model artifacts and test dataset stored in the repository. No external dataset download is required at runtime.

---

## j. Project Structure

```text
ML_01/
└── ML/
    ├── app.py                         # Streamlit dashboard application
    ├── requirements.txt               # Python dependencies
    ├── test_data.csv                  # Bundled 10,000-row held-out test dataset
    ├── logo.png                       # Dashboard logo
    ├── .gitignore
    ├── README.md
    │
    ├── model/
    │   ├── train_models.py             # Model training and evaluation
    │   ├── preprocessing.py            # Feature preprocessing pipeline
    │   ├── preprocessor.pkl            # Fitted preprocessing pipeline
    │   ├── logistic_regression.pkl
    │   ├── decision_tree.pkl
    │   ├── knn.pkl
    │   ├── naive_bayes.pkl
    │   ├── random_forest.pkl
    │   └── metrics_comparison.csv      # Final model comparison
    │
    └── notebooks/
        └── EDA_and_training.ipynb      # Exploratory analysis and experiments
```

---

## k. Application Features

The Streamlit dashboard provides:

- Test CSV upload
- Pre-trained model selection
- Dataset overview
- Current test-data performance metrics
- Model comparison
- Confusion matrix
- ROC curve and AUC
- Classification report
- Target-class distribution
- Feature statistics
- Logistic Regression coefficient analysis
- Decision Tree feature importance
- Random Forest feature importance

---

## l. Key Conclusion

The final experiment demonstrates that **the model with the highest accuracy is not necessarily the best model for an imbalanced classification problem**.

Although K-Nearest Neighbors achieved the highest accuracy at **85.19%**, its Recall was only **20.96%** and MCC was **0.2288**.

Logistic Regression achieved lower accuracy (**74.19%**) but substantially stronger:

- AUC: **0.8345**
- Recall: **77.82%**
- F1 Score: **0.4565**
- MCC: **0.3771**

Therefore, **Logistic Regression was selected as the overall winner for this experiment**, based on the combined evaluation of discrimination, minority-class detection, balance between precision and recall, and overall binary classification quality.

---

## m. Disclaimer

This project is intended for **educational and machine-learning demonstration purposes only**.

The predictions and risk classifications produced by this application are not medical diagnoses and should not be used as a substitute for professional medical advice, clinical evaluation, or treatment decisions.
