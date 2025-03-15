from sklearn.preprocessing import LabelEncoder


def label_encode_categorical_features(df):
    df = df.copy()

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(categorical_cols.tolist())
    print(len(categorical_cols.tolist()))


    # Apply Label Encoding to categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df, categorical_cols