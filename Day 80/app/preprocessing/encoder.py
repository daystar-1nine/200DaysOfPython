from sklearn.preprocessing import OneHotEncoder

def create_categorical_encoder() -> OneHotEncoder:
    """Create a one-hot encoder for categorical features."""
    return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
