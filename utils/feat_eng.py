import seaborn as sns
import matplotlib.pyplot as plt

def engineer_features(df, save_path=None):
    """
    Perform feature engineering by creating new combined features and analyzing correlation.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        save_path (str, optional): The path to save the processed DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    # remove the Id column
    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)
    else:
        print("Column 'Id' does not exist in the DataFrame.")

    # Example: Combine area-related features
    df['TotalLivingArea'] = df['TotalBsmtSF'] + df['GrLivArea']
    df['TotalFloorArea'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalOutdoorArea'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

    # Example: Combine age-related features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # Example: Combine bathroom-related features
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

    # Example: Combine quality-related features
    df['OverallScore'] = df['OverallQual'] + df['OverallCond']

    # # Example: Combine garage-related features
    # df['GarageSizeScore'] = df['GarageCars'] * df['GarageArea']

    # Combine 1stFlrSF, 2ndFlrSF, and TotalBsmtSF to create a new feature TotalSF.
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']

    # Select the new features and SalePrice for correlation analysis
    features_to_analyze = [
        'TotalLivingArea', 'TotalFloorArea', 'TotalOutdoorArea', 'HouseAge', 'RemodAge', 
        'TotalBathrooms', 'OverallScore', 'TotalSF', 'SalePrice'
    ]

    # Calculate the correlation matrix
    correlation_matrix = df[features_to_analyze].corr()

    # Extract correlation with SalePrice
    correlation_with_saleprice = correlation_matrix['SalePrice'].sort_values(ascending=False)

    # Display the correlation table
    print("Correlation with SalePrice:")
    print(correlation_with_saleprice)

    return df



#Function to get correlation map of the data
def get_correlation_map(df):
    plt.figure(figsize=(12, 8))
    # Select only numerical features for correlation analysis
    numerical_data = df.select_dtypes(include=['number'])

    sns.heatmap(numerical_data.corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Matrix")
    plt.show()

# function to get correlation table in reation with SalePrice only
def get_correlation_table(df):
    # Set display options to show all rows and columns

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Extract correlations with the target variable 'SalePrice'
    saleprice_correlations = correlation_matrix['SalePrice']
    
    # pd.set_option('display.max_rows', None)

    # Print the table
    return saleprice_correlations

def get_full_correlation_table(df):
    # Calculate the full correlation matrix
    correlation_matrix = df.corr()

    # Return the full correlation matrix
    return correlation_matrix


# drop columns with NaN values
def drop_nan_columns(df):

    columns_to_drop = [
    'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    ]
    
    # Filter out columns that don't exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    # Drop the columns
    df = df.drop(columns=existing_columns, inplace=False)

    # Verify the columns have been dropped
    print("Columns after dropping:")
    return df


def checking_correlation_vals(df):
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for val in vals:
        # Get features with correlation >= val
        features = abs(df.corr()['SalePrice'])[abs(df.corr()['SalePrice']) >= val].index.tolist()
        
        # Drop 'SalePrice' from the features list (if present)
        if 'SalePrice' in features:
            features.remove('SalePrice')
        
        # Drop 'SalePrice' from the DataFrame and select the remaining features
        x = df.drop(columns='SalePrice')
        x = x[features]
        
        print(f"Features with correlation >= {val}: {len(features)}")
        print(features)
        
        
# Function to select features based on correlation with the target variable

def feature_selection(df, target_column, correlation_threshold=0.3, handle_missing=True):
    """
    Selects features based on their correlation with the target variable.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features and the target variable.
    - target_column (str): The name of the target variable column.
    - correlation_threshold (float): The minimum absolute correlation value for feature selection.
    - handle_missing (bool): Whether to handle missing values in the selected features.

    Returns:
    - df_selected (pd.DataFrame): A DataFrame containing only the selected features and the target variable.
    - selected_features (list): A list of the selected feature names.
    """
    
    # Step 1: Calculate correlations with the target variable
    correlations = df.corr()[target_column]
    
    # Step 2: Select features with absolute correlation >= threshold
    selected_features = correlations[abs(correlations) >= correlation_threshold].index.tolist()
    
    # Step 3: Remove the target variable from the selected features (if present)
    if target_column in selected_features:
        selected_features.remove(target_column)
    
    # Step 4: Create a new DataFrame with only the selected features and the target variable
    df_selected = df[selected_features + [target_column]]
    
    # Step 5: Handle missing values (if enabled)
    if handle_missing:
        df_selected = df_selected.dropna()  # Drop rows with missing values
    
    all_features = df.columns.tolist()
    removed_features = [feature for feature in all_features if feature not in selected_features]

  
    # Step 6: Print summary
    print(f"Selected {len(selected_features)} features with correlation >= {correlation_threshold}:")
    print(selected_features)
    
    # print(f"\nRemoved Features {len(removed_features)}:")
    # print(removed_features)
    
    return df_selected, selected_features