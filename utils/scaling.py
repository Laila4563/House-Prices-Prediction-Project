import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_features(df, numerical_features, standardize=True, minmax=True, robust=True):
    """
    Scales numerical features based on the selected scaling techniques.

    Parameters:
        df (pd.DataFrame): The input preprocessed DataFrame.
        numerical_features (list): List of numerical feature column names.
        standardize (bool): Apply StandardScaler to normally distributed features.
        minmax (bool): Apply MinMaxScaler to skewed features.
        robust (bool): Apply RobustScaler to features with outliers.

    Returns:
        pd.DataFrame: Scaled DataFrame with categorical features unchanged.
    """

    binary_features = [col for col in df.columns if df[col].nunique() == 2]
    numerical_features = [col for col in numerical_features if col not in binary_features]

    scaled_df = df.copy()

    standard_features = []
    minmax_features = []
    robust_features = []

    # Identify features for each scaling method
    for col in numerical_features:
        skewness = df[col].skew()

        if abs(skewness) < 0.5:  # Normally distributed features
            standard_features.append(col)
        elif abs(skewness) > 1:  # Highly skewed features
            minmax_features.append(col)
        else:  # Moderately skewed (potentially with outliers)
            robust_features.append(col)

    # Apply StandardScaler
    if standardize and standard_features:
        scaler = StandardScaler()
        scaled_df[standard_features] = scaler.fit_transform(df[standard_features])

    # Apply MinMaxScaler
    if minmax and minmax_features:
        scaler = MinMaxScaler()
        scaled_df[minmax_features] = scaler.fit_transform(df[minmax_features])

    # Apply RobustScaler
    if robust and robust_features:
        scaler = RobustScaler()
        scaled_df[robust_features] = scaler.fit_transform(df[robust_features])

    return scaled_df