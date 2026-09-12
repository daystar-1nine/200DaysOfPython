import pandas as pd
from app.evaluation import calculate_rmse, calculate_mae, calculate_r2, calculate_adjusted_r2
from app.cross_validation import run_cross_validation

def compare_models(models_dict: dict, X_train, y_train, X_test, y_test, cv=5) -> pd.DataFrame:
    results = []
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = calculate_rmse(y_train, train_pred)
        test_rmse = calculate_rmse(y_test, test_pred)
        
        train_r2 = calculate_r2(y_train, train_pred)
        test_r2 = calculate_r2(y_test, test_pred)
        
        test_mae = calculate_mae(y_test, test_pred)
        
        n_features = X_train.shape[1]
        adj_r2 = calculate_adjusted_r2(test_r2, len(y_test), n_features)
        
        if name != "Baseline":
            cv_res = run_cross_validation(model, X_train, y_train, cv=cv)
            cv_rmse = cv_res["mean_cv_rmse"]
        else:
            cv_rmse = train_rmse
            
        results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "Test_RMSE": test_rmse,
            "Train_RMSE": train_rmse,
            "MAE": test_mae,
            "R2": test_r2,
            "Adjusted_R2": adj_r2
        })
        
    return pd.DataFrame(results)

def detect_overfitting(train_score, test_score, threshold=0.15) -> bool:
    if train_score == 0:
        return test_score > 0
    return ((test_score - train_score) / train_score > threshold) or ((train_score - test_score) > threshold)

def select_best_model(comparison_df: pd.DataFrame, metric="CV_RMSE") -> str:
    if metric in comparison_df.columns:
        if metric in ["R2", "Adjusted_R2"]:
            return comparison_df.loc[comparison_df[metric].idxmax(), "Model"]
        return comparison_df.loc[comparison_df[metric].idxmin(), "Model"]
    return comparison_df["Model"].iloc[0]
