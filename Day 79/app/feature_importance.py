from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def extract_feature_names(pipeline: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Extract transformed feature names from ColumnTransformer inside pipeline."""
    preprocessor = pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_names

def compute_mdi_importance(
    pipeline: Pipeline, 
    numeric_features: List[str], 
    categorical_features: List[str]
) -> pd.DataFrame:
    """Compute Mean Decrease in Impurity (MDI) feature importances with standard deviation across trees."""
    rf = pipeline.named_steps['classifier']
    feature_names = extract_feature_names(pipeline, numeric_features, categorical_features)
    
    importances = rf.feature_importances_
    tree_importances = np.array([tree.feature_importances_ for tree in rf.estimators_])
    std_importances = np.std(tree_importances, axis=0)
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Std': std_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return df_imp
