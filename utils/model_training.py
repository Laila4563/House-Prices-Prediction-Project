import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
import joblib
import os
import pickle

# Directory to store trained models
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, model_name):
    """
    Saves the trained model to a specified directory.
    """
    file_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    # Open the file in binary write mode
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {file_path}")
    

def load_model(model_name):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(PROJECT_ROOT, "models", f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    return joblib.load(model_path)


def split_data(df, target_column='SalePrice', test_size=0.2, val_size=0.5, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and the target variable.
        target_column (str): The name of the target variable column.
        test_size (float): Proportion of the dataset to be used for testing and validation.
        val_size (float): Proportion of the remaining data to be used for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Define features (X) and target variable (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 1: Split the data into training (80%) and temporary (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 2: Split the temporary set into validation (50%) and test (50%) sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_linear_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Trains a Linear Regression model using the normal equation.
    Prints model performance and returns a dictionary of metrics.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(model.coef_.shape)
    
    save_model(model, "linear_regression_model")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print formatted results
    print("\nLinear Regression Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # return {
    #     "Training R²": train_r2, "Validation R²": val_r2, "Test R²": test_r2,
    #     "Training MAE": train_mae, "Validation MAE": val_mae, "Test MAE": test_mae,
    # }


def plot_linear_regression(X_train, X_val, y_train, y_val, learning_rate=0.001, epochs=1000):
    """
    Trains a Linear Regression model using Gradient Descent and plots MSE over epochs.
    """
    m, n = X_train.shape
    theta = np.zeros(n)

    train_mse_history = []
    val_mse_history = []

    for epoch in range(epochs):
        y_train_pred = X_train.dot(theta)
        y_val_pred = X_val.dot(theta)

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)

        gradients = (1 / m) * X_train.T.dot(y_train_pred - y_train)
        theta -= learning_rate * gradients

    # Plot MSE over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_mse_history, label='Training MSE', color='blue')
    plt.plot(range(epochs), val_mse_history, label='Validation MSE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Linear Regression - MSE over Epochs')
    plt.legend()
    plt.show()

    # Print formatted results
    print(f"\nFinal Training MSE: {train_mse_history[-1]:.4f}")
    print(f"Final Validation MSE: {val_mse_history[-1]:.4f}")

    # return train_mse_history[-1], val_mse_history[-1]


def train_polynomial_regression(X_train, X_val, X_test, y_train, y_val, y_test, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)
    
    # Define the path to the parent folder's models directory
    models_dir = os.path.join(os.path.dirname(os.getcwd()), "models")
    
    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the polynomial transformer in the parent folder's models directory
    joblib.dump(poly, os.path.join(models_dir, f"polynomial_transformer_degree_{degree}.pkl"))


    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    print(model.coef_.shape)
    
    save_model(model, f"polynomial_regression_model_degree_{degree}")

    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    y_test_pred = model.predict(X_test_poly)

    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print formatted results
    print(f"\nPolynomial Regression (Degree={degree}) Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # return {
    #     "Training R²": train_r2, "Validation R²": val_r2, "Test R²": test_r2,
    #     "Training MAE": train_mae, "Validation MAE": val_mae, "Test MAE": test_mae,
    # }


def plot_polynomial_regression(X_train, X_val, y_train, y_val, degree=2, learning_rate=0.001, epochs=1000):
    """
    Trains a Polynomial Regression model using Gradient Descent.

    Returns:
        tuple: Final training and validation MSE along with the MSE history.
    """
    """
    Trains a Polynomial Regression model using Gradient Descent and plots MSE over epochs.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    m, n = X_train_poly.shape
    theta = np.zeros(n)

    train_mse_history = []
    val_mse_history = []

    for epoch in range(epochs):
        y_train_pred = X_train_poly.dot(theta)
        y_val_pred = X_val_poly.dot(theta)

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)

        gradients = (1 / m) * X_train_poly.T.dot(y_train_pred - y_train)
        theta -= learning_rate * gradients

    # Plot MSE over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_mse_history, label='Training MSE', color='blue')
    plt.plot(range(epochs), val_mse_history, label='Validation MSE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Polynomial Regression (Degree={degree}) - MSE over Epochs')
    plt.legend()
    plt.show()

    # Print formatted results
    print(f"\nFinal Training MSE: {train_mse_history[-1]:.4f}")
    print(f"Final Validation MSE: {val_mse_history[-1]:.4f}")

    # return train_mse_history[-1], val_mse_history[-1]


def train_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test, degree=2, alpha=0.5):
    """
    Trains a Polynomial Regression model with Lasso regularization using the normal equation.
    Prints model performance and returns a dictionary of metrics.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    print(model.coef_.shape)
    
    save_model(model, f"lasso_regression_model")

    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    y_test_pred = model.predict(X_test_poly)

    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print formatted results
    print(f"\nPolynomial Regression with Lasso (Degree={degree}, Alpha={alpha}) Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # return {
    #     "Training R²": train_r2, "Validation R²": val_r2, "Test R²": test_r2,
    #     "Training MAE": train_mae, "Validation MAE": val_mae, "Test MAE": test_mae,
    # }


def plot_lasso(X_train, X_val, y_train, y_val, degree=2, learning_rate=0.001, epochs=1000, alpha=0.1):
    """
    Trains a Polynomial Regression model with Lasso (L1) regularization using Gradient Descent.
    Plots MSE over epochs and prints the final loss values.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    m, n = X_train_poly.shape
    theta = np.zeros(n)

    train_mse_history = []
    val_mse_history = []

    for epoch in range(epochs):
        # Compute predictions
        y_train_pred = X_train_poly.dot(theta)
        y_val_pred = X_val_poly.dot(theta)

        # Compute loss (MSE)
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)

        # Compute gradients
        gradients = (1 / m) * X_train_poly.T.dot(y_train_pred - y_train)

        # Apply L1 Regularization (subgradient method)
        theta -= learning_rate * (gradients + alpha * np.sign(theta))

    # Plot MSE over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_mse_history, label='Training MSE', color='blue')
    plt.plot(range(epochs), val_mse_history, label='Validation MSE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Lasso Polynomial Regression (Degree={degree}, Alpha={alpha}) - MSE over Epochs')
    plt.legend()
    plt.show()

    # Print final MSE values
    print(f"\nFinal Training MSE: {train_mse_history[-1]:.4f}")
    print(f"Final Validation MSE: {val_mse_history[-1]:.4f}")

    # return train_mse_history[-1], val_mse_history[-1]


def train_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test, degree=2, alpha=0.5):
    """
    Trains a Polynomial Regression model with Ridge regularization.
    Prints model performance and returns a dictionary of metrics.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    model = Ridge(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    print(model.coef_.shape)
    
    save_model(model, f"ridge_regression_model")

    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    y_test_pred = model.predict(X_test_poly)

    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print formatted results
    print(f"\nPolynomial Regression with Ridge (Degree={degree}, Alpha={alpha}) Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # return train_r2, val_r2, test_r2, train_mae, val_mae, test_mae


def plot_ridge(X_train, X_val, y_train, y_val, degree=2, learning_rate=0.001, epochs=1000, alpha=0.1):
    """
    Trains a Polynomial Regression model with Ridge (L2) regularization using Gradient Descent.
    Plots MSE over epochs and prints the final loss values.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    m, n = X_train_poly.shape
    theta = np.zeros(n)

    train_mse_history = []
    val_mse_history = []

    for epoch in range(epochs):
        # Compute predictions
        y_train_pred = X_train_poly.dot(theta)
        y_val_pred = X_val_poly.dot(theta)

        # Compute loss (MSE)
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)

        # Compute gradients
        gradients = (1 / m) * X_train_poly.T.dot(y_train_pred - y_train)

        # Apply L2 Regularization
        theta -= learning_rate * (gradients + alpha * theta)

    # Plot MSE over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_mse_history, label='Training MSE', color='blue')
    plt.plot(range(epochs), val_mse_history, label='Validation MSE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Ridge Polynomial Regression (Degree={degree}, Alpha={alpha}) - MSE over Epochs')
    plt.legend()
    plt.show()

    # Print final MSE values
    print(f"\nFinal Training MSE: {train_mse_history[-1]:.4f}")
    print(f"Final Validation MSE: {val_mse_history[-1]:.4f}")

    # return train_mse_history[-1], val_mse_history[-1]


def train_elastic_net_regression(X_train, X_val, X_test, y_train, y_val, y_test, degree=2, alpha=0.5, l1_ratio=0.5):
    """
    Trains a Polynomial Regression model with Elastic Net regularization.
    Prints model performance and returns individual metrics.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    print(model.coef_.shape)
    
    save_model(model, f"elastic_net_regression_model")

    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    y_test_pred = model.predict(X_test_poly)

    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print formatted results
    print(f"\nPolynomial Regression with Elastic Net (Degree={degree}, Alpha={alpha}, L1 Ratio={l1_ratio}) Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # return train_r2, val_r2, test_r2, train_mae, val_mae, test_mae


def tune_ridge_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Create a pipeline for Polynomial Features + Ridge Model
    ridge_pipeline = make_pipeline(PolynomialFeatures(), Ridge())

    # Define hyperparameter grid
    param_grid = {
        'polynomialfeatures__degree': [1, 2, 3],  # Tune polynomial degree
        'ridge__alpha': [0.01, 0.1, 1.0, 10, 100]  # Tune Ridge alpha (regularization strength)
    }

    # Initialize GridSearchCV
    ridge_grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

    # Fit the model on training data
    ridge_grid_search.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    best_ridge_model = ridge_grid_search.best_estimator_
    
    print(best_ridge_model.named_steps['ridge'].coef_.shape)

    save_model(best_ridge_model, "ridge_tuned_model")

    # Make predictions with the best model
    y_train_pred = best_ridge_model.predict(X_train)
    y_val_pred = best_ridge_model.predict(X_val)
    y_test_pred = best_ridge_model.predict(X_test)

    # Evaluate the tuned model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print tuned model results
    print("Best Hyperparameters for Ridge:", ridge_grid_search.best_params_)
    print("Tuned Polynomial Regression with Ridge Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")


    # return ridge_grid_search.best_params_, (train_r2, val_r2), (train_mae, val_mae)


def tune_elastic_net_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Tunes ElasticNet regression model with polynomial features using RandomizedSearchCV."""
    # Define polynomial features
    degree = 2  # Change degree as needed
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    # Define hyperparameter search space
    param_distributions = {
        'alpha': np.logspace(-3, 2, 50),  # Wider range: 0.001 to 100
        'l1_ratio': np.linspace(0, 1, 50)  # More fine-grained search for L1/L2 mix
    }

    # Initialize Elastic Net model
    elastic_net = ElasticNet(random_state=42, max_iter=5000)

    # Perform Randomized Search with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        elastic_net, param_distributions, 
        n_iter=30,  # Randomly test 30 combinations
        scoring='neg_mean_absolute_error',  # Minimize MAE
        cv=5,  # 5-fold cross-validation
        random_state=42, 
        n_jobs=-1  # Use all CPU cores
    )
    random_search.fit(X_train_poly, y_train)

    # Get best hyperparameters
    best_alpha = random_search.best_params_['alpha']
    best_l1_ratio = random_search.best_params_['l1_ratio']
    print(f"Best Alpha: {best_alpha:.4f}, Best L1 Ratio: {best_l1_ratio:.4f}")

    # Train the best model using the optimal hyperparameters
    best_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42, max_iter=5000)
    best_model.fit(X_train_poly, y_train)
    
    print(best_model.coef_.shape)

    save_model(best_model, "elastic_net_tuned_model")

    # Make predictions
    y_train_pred = best_model.predict(X_train_poly)
    y_val_pred = best_model.predict(X_val_poly)
    y_test_pred = best_model.predict(X_test_poly)

    # Evaluate the model
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print results
    print("Elastic Net with Polynomial Features (Tuned) Model Performance:")
    print(f"Training R² Score: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"Validation R² Score: {val_r2:.4f}, MAE: {val_mae:.2f}")
    print(f"Test R² Score: {test_r2:.4f}, MAE: {test_mae:.2f}")
    
    # return random_search.best_params_, (train_r2, val_r2), (train_mae, val_mae)
