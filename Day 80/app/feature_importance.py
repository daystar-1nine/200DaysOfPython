from typing import List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def extract_pipeline_feature_names(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """Extract processed column names after ColumnTransformer."""
    preprocessor = pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_names

def compute_tree_feature_importance(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str]
) -> pd.DataFrame:
    """Compute MDI feature importance with standard deviation across ensemble trees."""
    classifier = pipeline.named_steps['classifier']
    feature_names = extract_pipeline_feature_names(pipeline, numeric_features, categorical_features)
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        if hasattr(classifier, 'estimators_'):
            tree_importances = np.array([tree.feature_importances_ for tree in classifier.estimators_])
            std_importances = np.std(tree_importances, axis=0)
        else:
            std_importances = np.zeros_like(importances)
            
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Std': std_importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df
    else:
        raise ValueError("Classifier does not support feature_importances_")
