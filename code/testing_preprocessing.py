

def testing_preprocess_data(df):
    X = df.copy(deep=True)

    # Drop columns from index 7 to 72
    # X.drop(X.columns[73:93], axis=1, inplace=True)
    # X.drop(X.columns[7:73], axis=1, inplace=True)

    # Drop the specified columns from X
    X.drop(columns=[
        'NEXT_DAY_CLOSEPRICE', 'DAILY_CLOSEPRICE_CHANGE', 'DAILY_CLOSEPRICE_CHANGE_PERCENT',
        'DAILY_CLOSEPRICE_DIRECTION',
        'DAILY_MIDPRICE', 'NEXT_DAY_MIDPRICE', 'DAILY_MIDPRICE_CHANGE', 'DAILY_MIDPRICE_CHANGE_PERCENT',
        'DAILY_MIDPRICE_DIRECTION',
        'Date'
    ], inplace=True)

    return X