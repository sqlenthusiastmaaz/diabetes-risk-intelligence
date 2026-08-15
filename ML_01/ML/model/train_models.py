import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from preprocessing import load_raw_data, preprocess_data, save_preprocessor

def main():
    print("Loading data...")
    df = load_raw_data()
    
    print("Splitting data into train and test sets...")
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["Diabetes_binary"], random_state=42
    )
    
    print("Preprocessing data...")
    X_train_scaled, y_train, preprocessor = preprocess_data(df_train, fit=True)
    X_test_scaled, y_test, _ = preprocess_data(df_test, fit=False, preprocessor=preprocessor)
    
    # Save the preprocessor
    save_preprocessor(preprocessor, "model/preprocessor.pkl")
    
    # Sample test_data.csv to keep it lightweight for Streamlit (approx 1500 rows)
    # We take it directly from df_test (unscaled) so the Streamlit app gets raw features
    df_test_sample, _ = train_test_split(
        df_test, train_size=1500, stratify=df_test["Diabetes_binary"], random_state=42
    )
    df_test_sample.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv with {len(df_test_sample)} rows")
    
    # Create subsampled training set for KNN (to keep pickle size small)
    knn_sample_size = 8000
    X_train_knn, _, y_train_knn, _ = train_test_split(
        X_train_scaled, y_train, train_size=knn_sample_size,
        stratify=y_train, random_state=42
    )
    print(f"KNN will train on {knn_sample_size} subsampled rows to keep model file small")
    
    # Define models with constrained parameters for smaller file sizes
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42, max_depth=12),
        "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, n_estimators=10, max_depth=12),
    }
    
    # Filenames for saving
    model_filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Gaussian Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
    }
    
    results = []
    
    os.makedirs("model", exist_ok=True)
    
    # Train and evaluate
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Use subsampled data for KNN to keep model file size small
        if name == "K-Nearest Neighbors":
            model.fit(X_train_knn, y_train_knn)
        else:
            model.fit(X_train_scaled, y_train)
        
        # Save model
        model_path = os.path.join("model", model_filenames[name])
        joblib.dump(model, model_path)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Saved {name} to {model_path} ({file_size_mb:.1f} MB)")
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        # Use predict_proba for AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Fallback for models that might not have predict_proba
            y_prob = y_pred
            
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        print(f"Metrics for {name}:")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
        
        results.append({
            "ML Model Name": name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "MCC": mcc
        })
        
    # Save results table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="MCC", ascending=False).reset_index(drop=True)
    
    print("\n--- Final Metrics Comparison ---")
    print(results_df)
    
    results_df.to_csv("model/metrics_comparison.csv", index=False)
    print("Saved metrics_comparison.csv")
    
    # Print total model sizes
    total_size = sum(
        os.path.getsize(os.path.join("model", f))
        for f in model_filenames.values()
        if os.path.exists(os.path.join("model", f))
    )
    print(f"\nTotal model file size: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
