import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from preprocessing import load_raw_data, preprocess_data, save_preprocessor


RANDOM_STATE = 42
WORKING_SAMPLE_SIZE = 50_000
TEST_SIZE = 0.20


def main():

    print("Loading full dataset...")
    df = load_raw_data()

    print(f"Original dataset size: {len(df):,} rows")

    # ---------------------------------------------------------
    # STEP 1: Create a stratified 50,000-row working dataset
    # ---------------------------------------------------------
    print(
        f"Creating stratified sample of {WORKING_SAMPLE_SIZE:,} rows..."
    )

    df_sample, _ = train_test_split(
        df,
        train_size=WORKING_SAMPLE_SIZE,
        stratify=df["Diabetes_binary"],
        random_state=RANDOM_STATE,
    )

    df_sample = df_sample.reset_index(drop=True)

    print(f"Working dataset size: {len(df_sample):,} rows")

    # ---------------------------------------------------------
    # STEP 2: 80/20 stratified train-test split
    # ---------------------------------------------------------
    print("Splitting working dataset into 80% train / 20% test...")

    df_train, df_test = train_test_split(
        df_sample,
        test_size=TEST_SIZE,
        stratify=df_sample["Diabetes_binary"],
        random_state=RANDOM_STATE,
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"Training rows: {len(df_train):,}")
    print(f"Test rows:     {len(df_test):,}")

    # ---------------------------------------------------------
    # STEP 3: Preprocessing
    # Fit ONLY on training data
    # ---------------------------------------------------------
    print("Preprocessing training data...")

    X_train_scaled, y_train, preprocessor = preprocess_data(
        df_train,
        fit=True
    )

    print("Transforming test data...")

    X_test_scaled, y_test, _ = preprocess_data(
        df_test,
        fit=False,
        preprocessor=preprocessor
    )

    # ---------------------------------------------------------
    # STEP 4: Save preprocessor
    # ---------------------------------------------------------
    os.makedirs("model", exist_ok=True)

    save_preprocessor(
        preprocessor,
        "model/preprocessor.pkl"
    )

    # ---------------------------------------------------------
    # STEP 5: Save SAME test set for Streamlit
    #
    # This is the exact same 10,000-row test set used
    # for the official evaluation below.
    # ---------------------------------------------------------
    df_test.to_csv(
        "test_data.csv",
        index=False
    )

    print(
        f"Saved test_data.csv with {len(df_test):,} rows"
    )

    # ---------------------------------------------------------
    # STEP 6: Define all models
    #
    # ALL models will use the SAME 40,000-row training set.
    # ---------------------------------------------------------
    models = {

        "Logistic Regression":
            LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_depth=12
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(
                n_jobs=-1
            ),

        "Gaussian Naive Bayes":
            GaussianNB(),

        "Random Forest":
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                n_estimators=10,
                max_depth=12
            ),
    }

    # ---------------------------------------------------------
    # STEP 7: Model filenames
    # ---------------------------------------------------------
    model_filenames = {

        "Logistic Regression":
            "logistic_regression.pkl",

        "Decision Tree":
            "decision_tree.pkl",

        "K-Nearest Neighbors":
            "knn.pkl",

        "Gaussian Naive Bayes":
            "naive_bayes.pkl",

        "Random Forest":
            "random_forest.pkl",
    }

    results = []

    # ---------------------------------------------------------
    # STEP 8: Train and evaluate ALL models
    # ---------------------------------------------------------
    for name, model in models.items():

        print(f"\n{'=' * 60}")
        print(f"Training {name}")
        print(f"{'=' * 60}")

        # IMPORTANT:
        # Every model gets the SAME 40,000 training rows.
        model.fit(
            X_train_scaled,
            y_train
        )

        # -----------------------------------------------------
        # Save model
        # -----------------------------------------------------
        model_path = os.path.join(
            "model",
            model_filenames[name]
        )

        joblib.dump(
            model,
            model_path
        )

        file_size_mb = (
            os.path.getsize(model_path)
            / (1024 * 1024)
        )

        print(
            f"Saved model: {model_path}"
        )

        print(
            f"Model size: {file_size_mb:.2f} MB"
        )

        # -----------------------------------------------------
        # Predictions on SAME 10,000-row test set
        # -----------------------------------------------------
        y_pred = model.predict(
            X_test_scaled
        )

        # Probability for AUC
        if hasattr(model, "predict_proba"):

            y_prob = model.predict_proba(
                X_test_scaled
            )[:, 1]

        else:

            y_prob = y_pred

        # -----------------------------------------------------
        # Calculate metrics
        # -----------------------------------------------------
        acc = accuracy_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        prec = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        print(
            f"Accuracy : {acc:.4f}"
        )
        print(
            f"AUC      : {auc:.4f}"
        )
        print(
            f"Precision: {prec:.4f}"
        )
        print(
            f"Recall   : {rec:.4f}"
        )
        print(
            f"F1 Score : {f1:.4f}"
        )
        print(
            f"MCC      : {mcc:.4f}"
        )

        results.append({

            "ML Model Name":
                name,

            "Accuracy":
                acc,

            "AUC":
                auc,

            "Precision":
                prec,

            "Recall":
                rec,

            "F1 Score":
                f1,

            "MCC":
                mcc,
        })

    # ---------------------------------------------------------
    # STEP 9: Save comparison table
    # ---------------------------------------------------------
    results_df = pd.DataFrame(
        results
    )

    # Sort by MCC because this is our primary
    # imbalance-aware overall comparison metric.
    results_df = (
        results_df
        .sort_values(
            by="MCC",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "model/metrics_comparison.csv",
        index=False
    )

    print(
        "\nSaved model/metrics_comparison.csv"
    )

    # ---------------------------------------------------------
    # STEP 10: Print total model size
    # ---------------------------------------------------------
    total_size = sum(

        os.path.getsize(
            os.path.join(
                "model",
                filename
            )
        )

        for filename
        in model_filenames.values()

        if os.path.exists(
            os.path.join(
                "model",
                filename
            )
        )
    )

    print(
        f"\nTotal model file size: "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()