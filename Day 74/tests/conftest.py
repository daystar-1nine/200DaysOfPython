import pytest, os, pandas as pd, numpy as np
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DAY74_DIR = os.path.dirname(TEST_DIR)

@pytest.fixture
def config():
    try: from app.config import AppConfig
    except: from config import AppConfig
    return AppConfig()

@pytest.fixture
def sample_df():
    # create a minimal 50-row DataFrame matching the schema
    np.random.seed(0)
    n = 50
    regions = ['North','South','East','West']
    cats = ['Electronics','Clothing','Food','Home','Sports']
    return pd.DataFrame({
        'Record_ID': [f'T-{i:03d}' for i in range(n)],
        'Date': ['2022-01-01'] * n,
        'TV_Spend': np.random.uniform(500,15000,n),
        'Digital_Spend': np.random.uniform(200,12000,n),
        'Radio_Spend': np.random.uniform(100,5000,n),
        'Discount': np.random.uniform(0,30,n),
        'Quantity': np.random.randint(10,500,n),
        'Region': np.random.choice(regions, n),
        'Category': np.random.choice(cats, n),
        'Competitor_Price': np.random.uniform(50,500,n),
        'Customer_Count': np.random.randint(100,5000,n),
        'Advertising_Spend': np.random.uniform(1000,30000,n),
        'Sales': np.random.uniform(5000,50000,n),
        'Profit': np.random.uniform(500,5000,n),
        'Month': np.random.randint(1,13,n),
        'Year': np.random.choice([2022,2023],n),
    })
