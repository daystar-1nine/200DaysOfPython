import pytest
import pandas as pd

def compare_models(results_dict):
    df = pd.DataFrame(results_dict).T
    df['Overfitting'] = df['Train_RMSE'] < df['Test_RMSE'] * 0.8  # dummy logic
    return df

def test_compare_models_returns_dataframe():
    res = {'ModelA': {'Train_RMSE': 1.0, 'Test_RMSE': 1.2}}
    df = compare_models(res)
    assert isinstance(df, pd.DataFrame)

def test_compare_models_has_all_models():
    res = {'ModelA': {'Train_RMSE': 1.0, 'Test_RMSE': 1.2},
           'ModelB': {'Train_RMSE': 0.8, 'Test_RMSE': 1.5}}
    df = compare_models(res)
    assert 'ModelA' in df.index
    assert 'ModelB' in df.index

def test_detect_overfitting_flag():
    res = {'ModelA': {'Train_RMSE': 1.0, 'Test_RMSE': 1.2},
           'ModelB': {'Train_RMSE': 0.5, 'Test_RMSE': 2.0}}
    df = compare_models(res)
    assert not df.loc['ModelA', 'Overfitting']
    assert df.loc['ModelB', 'Overfitting']

def test_select_best_model_picks_lowest_rmse():
    res = {'ModelA': {'Train_RMSE': 1.0, 'Test_RMSE': 1.2},
           'ModelB': {'Train_RMSE': 0.8, 'Test_RMSE': 1.5}}
    df = compare_models(res)
    best_model = df['Test_RMSE'].idxmin()
    assert best_model == 'ModelA'
