import numpy as np

def training_preprocess_data(df):
    if df.isna().sum().sum() > 0 or df.isin([float('inf'), float('-inf')]).sum().sum() > 0:
        df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()


    # Create target variables before dropping columns
    y_classifier = (np.sign(df['DAILY_MIDPRICE_CHANGE']) >= 0).astype(int)
    y_regressor = df['DAILY_MIDPRICE_CHANGE']

    # Drop columns from index 7 to 72
    #df = df.drop(df.columns[73:93], axis=1)
    #df = df.drop(df.columns[7:73], axis=1)

    # Drop specific columns
    columns_to_drop = [
        'NEXT_DAY_CLOSEPRICE', 'DAILY_CLOSEPRICE_CHANGE', 'DAILY_CLOSEPRICE_CHANGE_PERCENT', 'DAILY_CLOSEPRICE_DIRECTION',
        'DAILY_MIDPRICE', 'NEXT_DAY_MIDPRICE', 'DAILY_MIDPRICE_CHANGE', 'DAILY_MIDPRICE_CHANGE_PERCENT', 'DAILY_MIDPRICE_DIRECTION',
        'Date',
    ]
    df = df.drop(columns=columns_to_drop)

    X = df

    return X, y_classifier, y_regressor