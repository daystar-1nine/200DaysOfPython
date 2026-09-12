import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Total_Ad_Spend'] = df['TV_Spend'] + df['Digital_Spend'] + df['Radio_Spend']
    df['Ad_Efficiency'] = df['Sales'] / df['Total_Ad_Spend'].replace(0, 1)
    df['Discount_Intensity'] = df['Discount'] * df['Quantity']
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
    else:
        df['Month'] = 1
        df['Year'] = 2023
        
    return df
