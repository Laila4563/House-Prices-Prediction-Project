import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

def treat_skewness(df, continuous_features, skew_threshold=0.5, mild_threshold=1.0, moderate_threshold=2.0):
    df = df.copy()
    
    # Get skewness values for all continuous features
    skewness = df[continuous_features].apply(skew)

    # Find features where skewness exceeds the threshold
    skewed_features = skewness[abs(skewness) > skew_threshold].index.tolist()
    print(f"Found {len(skewed_features)} skewed continuous features: {skewed_features}")

    # Store details of the transformations (for future reference if needed)
    transformation_details = {}

    for feature in skewed_features:
        feature_skew = skewness[feature]
        original_feature = df[feature].copy()  # Save original values in case we need to revert

        # If the feature is negatively skewed, reflect the data to make it positive
        if feature_skew < 0:
            df[feature] = df[feature].max() + 1 - df[feature]
            feature_skew = abs(feature_skew)  # Treat it as positive for transformation decisions
            print(f"{feature} is negatively skewed. Data reflected.")

        # Choose the right transformation based on how severe the skewness is
        if skew_threshold < feature_skew <= mild_threshold:
            # Mild skewness → Square root transformation
            if df[feature].min() < 0:
                shift_value = abs(df[feature].min()) + 1
                df[feature] = np.sqrt(df[feature] + shift_value)
                transformation_type = f"Square Root (shifted by {shift_value})"
            else:
                df[feature] = np.sqrt(df[feature])
                transformation_type = "Square Root"

        elif mild_threshold < feature_skew <= moderate_threshold:
            # Moderate skewness → Log transformation
            if df[feature].min() < 0:
                shift_value = abs(df[feature].min()) + 1
                df[feature] = np.log1p(df[feature] + shift_value)
                transformation_type = f"Log(1 + x) (shifted by {shift_value})"
            else:
                df[feature] = np.log1p(df[feature])
                transformation_type = "Log(1 + x)"

        else:
            # Severe skewness → Use advanced transformers
            if feature_skew > 5:
                qt = QuantileTransformer(output_distribution='normal')
                df[feature] = qt.fit_transform(df[[feature]])
                transformation_type = "Quantile Transformation"
                transformation_details[feature] = qt
            else:
                pt = PowerTransformer(method='yeo-johnson')
                df[feature] = pt.fit_transform(df[[feature]])
                transformation_type = "Yeo-Johnson"
                transformation_details[feature] = pt

        # Check if the transformation actually reduced the skewness
        new_skew = skew(df[feature])
        if abs(new_skew) >= abs(feature_skew):
            print(f"{feature} - No improvement, reverting...")
            df[feature] = original_feature
            continue

        print(f"{feature} transformed using {transformation_type}")

    return df, skewed_features, transformation_details
