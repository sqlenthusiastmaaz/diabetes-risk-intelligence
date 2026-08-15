# Diabetes Risk Intelligence

## a. Problem Statement
Diabetes is one of the most prevalent chronic diseases worldwide. Early detection of diabetes or prediabetes is critical for timely intervention, lifestyle changes, and preventing severe complications such as heart disease or vision loss. This project frames a real-world healthcare screening tool: predicting diabetes risk based on accessible lifestyle and health survey indicators, empowering healthcare providers and individuals with early risk assessment.

## b. Dataset Description
- **Source**: Kaggle ([Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)), derived from the CDC's BRFSS 2015 survey.
- **Size**: 253,680 rows and 22 columns.
- **Features**: 21 numeric/binary features covering lifestyle choices (e.g., smoking, physical activity, diet) and health metrics (e.g., BMI, high blood pressure, cholesterol).
- **Target Variable**: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).
- **Class Imbalance Note**: The dataset is heavily imbalanced, with ~86% of respondents classified as non-diabetic. This makes standard accuracy misleading, so we emphasize metrics like the Matthews Correlation Coefficient (MCC), Area Under the Curve (AUC), and Recall.

## c. GitHub Repository Link
[https://github.com/sqlenthusiastmaaz/diabetes-risk-intelligence](https://github.com/sqlenthusiastmaaz/diabetes-risk-intelligence)

## d. Models Used
Below is the performance comparison of the 5 models evaluated on the 20% held-out test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.7437 | 0.8183 | 0.3187 | 0.7383 | 0.4452 | 0.3576 |
| **Logistic Regression** | 0.7315 | 0.8196 | 0.3107 | 0.7611 | 0.4413 | 0.3563 |
| **Decision Tree** | 0.7019 | 0.7864 | 0.2869 | 0.7672 | 0.4176 | 0.3284 |
| **Gaussian Naive Bayes** | 0.7722 | 0.7803 | 0.3198 | 0.5639 | 0.4082 | 0.2974 |
| **K-Nearest Neighbors** | 0.8453 | 0.7037 | 0.3774 | 0.1702 | 0.2346 | 0.1780 |

## e. Observations Table

| ML Model Name | Observation |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest AUC (0.8196) and the second-best MCC (0.3563). Using `class_weight="balanced"` pushed recall to 76%, making it ideal for medical screening where missing a diabetic case is costlier than a false alarm. |
| **Decision Tree** | With `max_depth=12` and balanced class weights, the depth-limited tree achieved good recall (76.7%) and a competitive AUC (0.7864). Constraining depth reduced overfitting compared to an unrestricted tree. |
| **K-Nearest Neighbors** | Achieved the highest raw accuracy (84.5%) but the poorest recall (17%) and MCC (0.178). KNN is sensitive to class imbalance and high dimensionality, predominantly predicting the majority class. |
| **Gaussian Naive Bayes** | Surprisingly decent performance with the conditional independence assumption. Balanced AUC (0.78) and moderate recall (56.4%), though precision remains limited. |
| **Random Forest** | Achieved the best MCC (0.3576) and strong AUC (0.8183) with only 10 depth-limited trees. The ensemble approach balances recall (73.8%) and precision (31.9%) effectively, providing the best overall discriminatory power. |
| **Overall Winner** | **Random Forest (Ensemble)**. With balanced class weights and constrained tree depth, Random Forest achieves the highest MCC (0.3576), best F1 (0.4452), and a near-best AUC (0.8183), providing the strongest overall balance between identifying diabetic cases and avoiding false alarms. |

## f. How to Run

### Local Setup
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <repository-name>
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the training script** (optional — pre-trained models are included):
   ```bash
   python model/train_models.py
   ```
4. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

### Streamlit Community Cloud Deployment
1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click "New app".
3. Select your repository, branch, and set the main file path to `app.py`.
4. Deploy. The app runs independently using bundled `test_data.csv` and `model/*.pkl` files — no external data downloads required.

## g. Project Structure
```
ML/
├── app.py                  # Streamlit dashboard application
├── requirements.txt        # Python dependencies
├── test_data.csv           # Bundled test dataset (1,500 rows)
├── logo.png                # Dashboard logo
├── .gitignore
├── model/
│   ├── train_models.py     # Training script
│   ├── preprocessing.py    # Feature preprocessing pipeline
│   ├── preprocessor.pkl    # Fitted StandardScaler
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── metrics_comparison.csv
└── notebooks/
    └── EDA_and_training.ipynb
```
