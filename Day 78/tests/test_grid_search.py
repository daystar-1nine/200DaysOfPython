from app.tuning.grid_search import tune_decision_tree
from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor
from app.feature_engineering import engineer_features

def test_tune_decision_tree(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    base_m = build_decision_tree_model(prep)
    
    grid = {
        "classifier__max_depth": [2, 3],
        "classifier__min_samples_split": [2, 5]
    }
    best_est, best_params, best_score, cv_df = tune_decision_tree(
        base_m, df_feat, sample_df["Churn"], param_grid=grid, n_splits=3
    )
    assert best_est is not None
    assert "classifier__max_depth" in best_params
    assert best_score >= 0.0
    assert not cv_df.empty
