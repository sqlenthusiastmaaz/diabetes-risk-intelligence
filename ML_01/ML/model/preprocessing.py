import pandas as pd
import kagglehub
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Any

def load_raw_data() -> pd.DataFrame:
    """
    Downloads the Diabetes Health Indicators Dataset via kagglehub and loads the 
    binary classification CSV into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    csv_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(
    df: pd.DataFrame, 
    fit: bool = True, 
    preprocessor: Optional[Any] = None
) -> Tuple[pd.DataFrame, pd.Series, Any]:
    """
    Separates features (X) and target (y), and applies standard scaling.
    
    Note on Class Imbalance:
        The dataset is heavily imbalanced (~86% non-diabetic). While we do not use 
        SMOTE here, we will handle this during model training by utilizing 
        `class_weight="balanced"` for supported algorithms (e.g., Logistic Regression, 
        Random Forest, Decision Tree).
        
    Args:
        df (pd.DataFrame): The raw dataframe.
        fit (bool): Whether to fit a new preprocessor or transform using an existing one.
        preprocessor (Optional[Any]): An existing fitted preprocessor to use if fit=False.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, Any]: Processed X, target y, and the preprocessor.
    """
    target_col = "Diabetes_binary"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if fit:
        # Scale all numeric/binary features
        # We use StandardScaler for all columns. Binary variables scaled will just shift/scale,
        # which is perfectly fine for models like Logistic Regression and KNN.
        numeric_features = X.columns.tolist()
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        X_scaled_array = preprocessor.fit_transform(X)
    else:
        if preprocessor is None:
            raise ValueError("A fitted preprocessor must be provided if fit=False")
        X_scaled_array = preprocessor.transform(X)
        
    # Convert back to DataFrame for interpretability / feature names
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    
    return X_scaled, y, preprocessor

def save_preprocessor(preprocessor: Any, path: str) -> None:
    """
    Saves the fitted preprocessor to disk using joblib.
    
    Args:
        preprocessor (Any): The fitted preprocessor.
        path (str): File path to save the preprocessor to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")

def load_preprocessor(path: str) -> Any:
    """
    Loads a saved preprocessor from disk.
    
    Args:
        path (str): File path to load the preprocessor from.
        
    Returns:
        Any: The loaded preprocessor.
    """
    return joblib.load(path)
