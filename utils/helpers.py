import pandas as pd

def read_file(path):
    """Read a file and return its contents."""
    return pd.read_csv(path)

def save_file(df, path):
    """Save a DataFrame to a file."""
    df.to_csv(path, index=False)