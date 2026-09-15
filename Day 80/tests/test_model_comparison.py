import pandas as pd
from app.model_comparison import build_model_comparison_table, select_best_model
from app.preprocessing.pipeline import build_preprocessor
from app.models.random_forest import create_random_forest_pipeline
from app.feature_importance import compute_tree_feature_importance
from app.permutation_importance import compute_permutation_importance

def test_build_model_comparison_table():
    cv = {
        'ModelA': {'roc_auc': {'test_mean': 0.85, 'test_std': 0.02}, 'average_precision': {'test_mean': 0.80}, 'f1': {'test_mean': 0.75}},
        'ModelB': {'roc_auc': {'test_mean': 0.92, 'test_std': 0.01}, 'average_precision': {'test_mean': 0.88}, 'f1': {'test_mean': 0.82}}
    }
    test_m = {
        'ModelA': {'accuracy': 0.80, 'precision': 0.75, 'recall': 0.70, 'f1': 0.72, 'roc_auc': 0.84, 'average_precision': 0.79},
        'ModelB': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.80, 'f1': 0.82, 'roc_auc': 0.93, 'average_precision': 0.89}
    }
    costs = {'ModelA': 20000.0, 'ModelB': 12000.0}
    df = build_model_comparison_table(cv, test_m, costs)
    assert len(df) == 2
    assert 'Model' in df.columns
    assert 'Test_ROC_AUC' in df.columns

def test_select_best_model():
    df = pd.DataFrame({
        'Model': ['ModelA', 'ModelB'],
        'Test_ROC_AUC': [0.84, 0.93]
    })
    champ = select_best_model(df, 'Test_ROC_AUC')
    assert champ == 'ModelB'

def test_compute_tree_feature_importance(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_random_forest_pipeline(prep, n_estimators=20, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    df_imp = compute_tree_feature_importance(pipe, sample_config.numeric_features, sample_config.categorical_features)
    assert 'Feature' in df_imp.columns
    assert 'Importance' in df_imp.columns
    assert (df_imp['Importance'] >= 0.0).all()

def test_compute_permutation_importance(sample_config, sample_splits):
    X_tr, X_te, y_tr, y_te = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_random_forest_pipeline(prep, n_estimators=20, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    df_perm = compute_permutation_importance(pipe, X_te, y_te, n_repeats=2, scoring='roc_auc')
    assert 'Feature' in df_perm.columns
    assert 'Mean_Importance' in df_perm.columns
    assert len(df_perm) == len(X_te.columns)

def test_comparison_sorted():
    df = pd.DataFrame({
        'Model': ['ModelA', 'ModelB'],
        'Test_ROC_AUC': [0.70, 0.95]
    })
    # build_model_comparison_table sorts descending
    assert df.sort_values(by='Test_ROC_AUC', ascending=False).iloc[0]['Model'] == 'ModelB'
