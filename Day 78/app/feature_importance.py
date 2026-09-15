import pandas as pd

def extract_feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    clf = pipeline.named_steps["classifier"]
    importances = clf.feature_importances_
    
    df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df
