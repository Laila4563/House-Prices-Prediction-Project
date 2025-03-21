import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Get the absolute path of the project root (one level up from 'pipeline/')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)  # Add project root to Python path

from utils.helpers import read_file
from utils.preprocessing import fix_logically_missing_values, clean_missing_values, handle_outliers_iqr
from utils.encoding import label_encode_categorical_features
from utils.skewness import treat_skewness
from utils.scaling import scale_features
from utils.feat_eng import engineer_features, drop_nan_columns, feature_selection
from utils.model_training import load_model

# Load the raw test data
df = read_file(os.path.join(PROJECT_ROOT, "data", "train.csv"))

if "Id" in df.columns:
    id_column = df["Id"]
    
# Extract the target variable (SalePrice)
y_test = df["SalePrice"]

# Apply the same preprocessing steps as training
df = fix_logically_missing_values(df)
df = clean_missing_values(df)

# List of numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df = handle_outliers_iqr(df, numerical_features)

# Handle categorical encoding
df, categorical_cols = label_encode_categorical_features(df)

# List of numerical features
numerical_features = [col for col in df if col not in categorical_cols and col != 'SalePrice' and col != 'Id']

# Treat skewness
df, skewed_features, transformation_details = treat_skewness(df, numerical_features)

# Scale features
df = scale_features(df, numerical_features)

# Feature engineering
df = engineer_features(df)

# Drop unnecessary columns
df = drop_nan_columns(df)

# Select features based on what was used in training
df, selected_features = feature_selection(df, target_column='SalePrice')

# Load trained models
linear_model = load_model("linear_regression_model")
lasso_model = load_model("lasso_regression_model")
ridge_model = load_model("ridge_regression_model")
elastic_net_model = load_model("elastic_net_regression_model")
ridge_tuned_model = load_model("ridge_tuned_model")
elastic_net_tuned_model = load_model("elastic_net_tuned_model")

# Load the trained polynomial transformer
models_dir = os.path.join(PROJECT_ROOT, "models")  
poly_transformer_path = os.path.join(models_dir, "polynomial_transformer_degree_2.pkl")
poly = joblib.load(poly_transformer_path)

# Transform the selected features using the polynomial transformer
df_poly = poly.transform(df[selected_features])

# Load polynomial regression model
polynomial_model = load_model("polynomial_regression_model_degree_2")

# Prepare DataFrame for storing results
results = pd.DataFrame()
results["Id"] = id_column  # Keep track of original row IDs

# Make predictions
models = {
    "Linear Regression": (linear_model, df[selected_features]),
    "Polynomial Regression": (polynomial_model, df_poly),
    "Lasso Regression": (lasso_model, df_poly),
    "Ridge Regression": (ridge_model, df_poly),
    "Elastic Net Regression": (elastic_net_model, df_poly),
    "Ridge Regression (Tuned)": (ridge_tuned_model, df[selected_features]),
    "Elastic Net Regression (Tuned)": (elastic_net_tuned_model, df_poly),
}

# Store metrics
metrics = []

for model_name, (model, X) in models.items():
    y_pred = model.predict(X)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Append results to the list
    metrics.append((model_name, r2, mae))
    
    print(f"\n=== {model_name} Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")

# Define the results directory
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics, columns=["Model", "R2 Score", "MAE"])
metrics_df.to_csv(os.path.join(results_dir, "model_performance.csv"), index=False)

print("\nPredictions and performance metrics saved successfully!")
print("Test pipeline executed successfully!")