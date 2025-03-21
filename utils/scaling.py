import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def is_normally_distributed(data, alpha=0.05):
    """
    Checks if the data is normally distributed using the Shapiro-Wilk test.
    """
    
    if data.nunique() < 3:
        return False
    
    skew_val = data.skew()
    kurtosis_val = data.kurtosis()

    if len(data) > 500:
        return abs(skew_val) < 0.5 and abs(kurtosis_val) < 3.5

    stat, p_value = shapiro(data.dropna())
    return p_value > alpha

def plot_distribution(df, feature):
    """
    Plots histogram and Q-Q plot to check feature normality.
    """
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], kde=True, bins=30, color='blue')
    plt.title(f"Histogram of {feature}")

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {feature}")

    plt.show()

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

    # Identify scaling method for each feature using Shapiro-Wilk test and skewness
    print("\n📊 Feature Scaling Decision Log:")
    for col in numerical_features:
        feature_data = df[col].dropna()  
        normal = is_normally_distributed(feature_data)
        skew_val = feature_data.skew()
        kurtosis_val = feature_data.kurtosis()
        outliers = ((feature_data > feature_data.quantile(0.99)) | (feature_data < feature_data.quantile(0.01))).sum()

        # plot_distribution(df, col)


        min_val = feature_data.min()
        max_val = feature_data.max()
        range_val = max_val - min_val
        print(f"Feature: {col}, Min: {min_val}, Max: {max_val}, Range: {range_val}")
        print(f"Feature: {col}, Skewness: {skew_val}, Kurtosis: {kurtosis_val}, Outliers: {outliers}")

        # Decision logic for scaling method
        reason = ""
        if range_val < 10:  # 🔹 Small range → MinMaxScaler
            minmax_features.append(col)
            reason = "Small range (<10), MinMaxScaler chosen."
        elif normal and outliers == 0:  # 🔹 Normal & No Outliers → StandardScaler
            standard_features.append(col)
            reason = "Normally distributed with no outliers, StandardScaler chosen."
        elif abs(skew_val) > 0.5 and abs(skew_val) < 1.5:  # 🔹 Moderate Skew → MinMaxScaler
            minmax_features.append(col)
            reason = f"Moderate skew (|{skew_val:.3f}| > 0.5), MinMaxScaler chosen."
        elif abs(kurtosis_val) > 3.5 or outliers > 10:  # 🔹 High Kurtosis or Many Outliers → RobustScaler
            robust_features.append(col)
            reason = f"High kurtosis (|{kurtosis_val:.3f}| > 3.5) or many outliers ({outliers}), RobustScaler chosen."
        else:
            standard_features.append(col)  # Default to StandardScaler
            reason = "Defaulting to StandardScaler."
            
        print(f"   → {reason}\n")

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
        
    print(f"StandardScaler applied to: {standard_features}")
    print(f"MinMaxScaler applied to: {minmax_features}")
    print(f"RobustScaler applied to: {robust_features}")

    return scaled_df