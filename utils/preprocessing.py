import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_unique_values(df):
    """
    Returns a dictionary containing unique values for each column in the DataFrame.
    Prints the unique values for each column as well.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        dict: A dictionary where keys are column names and values are arrays of unique values.
    """
    unique_values = {col: df[col].unique() for col in df.columns}
    
    for col, values in unique_values.items():
        print(f"{col}: {values}\n")
    


#Grouping the related features to handle data logically
def fix_logically_missing_values(df):
    df = df.copy()  # Avoid modifying the original data

    feature_groups = {
        "Garage": {
            "primary": "GarageType",
            "related_categorical": ["GarageFinish", "GarageQual", "GarageCond"],
            "related_numerical": ["GarageYrBlt", "GarageCars", "GarageArea"]
        },
        "Basement": {
            "primary": "BsmtQual",
            "related_categorical": ["BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
            "related_numerical": ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
        },
        "Masonry": {
            "primary": "MasVnrType",
            "related_categorical": [],
            "related_numerical": ["MasVnrArea"]
        },
        "Fireplace": {
            "primary": "Fireplaces",
            "related_categorical": ["FireplaceQu"],
            "related_numerical": []
        },
        "Pool": {
            "primary": "PoolArea",
            "related_categorical": ["PoolQC"],
            "related_numerical": []
        },
        "Fence": {
            "primary": "Fence",
            "related_categorical": [],
            "related_numerical": []
        },
        "Alley": {
            "primary": "Alley",
            "related_categorical": [],
            "related_numerical": []
        },
        "Miscellaneous": {
            "primary": "MiscFeature",
            "related_categorical": [],
            "related_numerical": ["MiscVal"]
        },
        "SecondFloor": {
            "primary": "2ndFlrSF",
            "related_categorical": [],
            "related_numerical": []
        }
    }

    for key, features in feature_groups.items():
        primary_col = features["primary"]
        categorical_cols = features["related_categorical"]
        numerical_cols = features["related_numerical"]

        # Ensure all columns exist to prevent KeyErrors
        if primary_col not in df.columns:
            continue

        # Create a mask for missing primary feature
        missing_mask = df[primary_col].isna()

        for col in categorical_cols:
            if col in df.columns:
                df.loc[missing_mask, col] = "Not Available"

        for col in numerical_cols:
            if col in df.columns:
                df.loc[missing_mask, col] = 0

    return df


def clean_missing_values(df):
    df = df.copy()

    # Handle categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df.loc[:, col] = df[col].replace(["None", "NA", np.nan], "Not Available")

    # Handle numerical columns
    for col in df.select_dtypes(include=['number']).columns:
        missing_percentage = df[col].isna().mean() * 100  #missing values' percentages

        if missing_percentage > 50:
            print(f"Dropping column '{col}' due to {missing_percentage:.2f}% missing values.")
            df.drop(columns=[col], inplace=True)
        elif missing_percentage > 0:
            median_value = df[col].median()
            print(f"Filling missing values in '{col}' with median: {median_value}")
            df.loc[:, col] = df[col].fillna(median_value)

    return df


def look_for_outliers(df):
    # Look for outliers:
    for col in df.select_dtypes(include='number').columns:
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()
        
        
# Function to handle outliers
def handle_outliers_iqr(df, column):
    """
    Handle outliers in a DataFrame column using the IQR method.
    - Caps outliers at the lower and upper bounds.
    """
    Q1 = df[column].quantile(0.25)  # 25th percentile
    Q3 = df[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers

    # Cap outliers at the lower and upper bounds
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    return df