import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

'''
0: Date
1: Open
2: High
3: Low
4: Close
5: Volume
6: Change
7: Wave
8: Prev. Day Open Interest
9: EFP Volume
10: EFS Volume
11: Block Volume
12: Last
13: Previous Day Open Interest
14: BB_UPPER
15: BB_MIDDLE
16: BB_LOWER
17: DEMA
18: EMA
19: HT_TRENDLINE
20: KAMA
21: MA
22: MAMA
23: FAMA
24: MAVP
25: MIDPOINT
26: MIDPRICE
27: SAR
28: SAREXT
29: SMA
30: T3
31: TEMA
32: TRIMA
33: WMA
34: ADX
35: ADXR
36: APO
37: AROON_down
38: AROON_up
39: AROONOSC
40: BOP
41: CCI
42: CMO
43: DX
44: MACD
45: MACD_signal
46: MACD_hist
47: MACDEXT
48: MACDEXT_signal
49: MACDEXT_hist
50: MACDFIX
51: MACDFIX_signal
52: MACDFIX_hist
53: MFI
54: MINUS_DI
55: MINUS_DM
56: MOM
57: PLUS_DI
58: PLUS_DM
59: PPO
60: ROC
61: ROCP
62: ROCR
63: ROCR100
64: RSI
65: STOCH_slowk
66: STOCH_slowd
67: STOCHF_fastk
68: STOCHF_fastd
69: STOCHRSI_fastk
70: STOCHRSI_fastd
71: TRIX
72: ULTOSC
73: WILLR
74: AD
75: ADOSC
76: OBV
77: ATR
78: NATR
79: TRANGE
80: HT_DCPERIOD
81: HT_DCPHASE
82: HT_PHASOR_inphase
83: HT_PHASOR_quadrature
84: HT_SINE_sine
85: HT_SINE_leadsine
86: HT_TRENDMODE
87: AVGPRICE
88: MEDPRICE
89: TYPPRICE
90: WCLPRICE
91: BETA
92: CORREL
93: LINEARREG
94: LINEARREG_ANGLE
95: LINEARREG_INTERCEPT
96: LINEARREG_SLOPE
97: STDDEV
98: TSF
99: VAR
100: DEMA_Trend
101: EMA_Trend
102: HT_TRENDLINE_Trend
103: KAMA_Trend
104: MA_Trend
105: MAMA_Trend
106: MAVP_Trend
107: MIDPOINT_Trend
108: MIDPRICE_Trend
109: SAR_Trend
110: SAREXT_Trend
111: SMA_Trend
112: T3_Trend
113: TEMA_Trend
114: TRIMA_Trend
115: WMA_Trend
116: ADX_Trend
117: ADXR_Trend
118: AROONOSC_Trend
119: DX_Trend
120: TRIX_Trend
121: DMI_Trend
122: AROON_Up_Trend
123: AROON_Down_Trend
124: PM_Uptrend
125: PM_Downtrend
126: ROC_Trend
127: ROCP_Trend
128: ROCR_Trend
129: ROCR100_Trend
130: ROC_Buy_Sell_Signal
131: ROCP_Buy_Sell_Signal
132: ROCR_Buy_Sell_Signal
133: ROCR100_Buy_Sell_Signal
134: APO_Buy_Sell_Signal
135: MACD_Buy_Sell_Signal
136: MACDEXT_Buy_Sell_Signal
137: MACDFIX_Buy_Sell_Signal
138: PPO_Buy_Sell_Signal
139: MOM_Buy_Sell_Signal
140: STOCH_Buy_Sell_Signal
141: STOCHF_Buy_Sell_Signal
142: STOCHRSI_Buy_Sell_Signal
143: ULTOSC_Buy_Sell_Signal
144: WILLR_Buy_Sell_Signal
145: BOP_Buy_Sell_Pressure
146: MFI_Buy_Sell_Pressure
147: AD_Buy_Sell_Pressure
148: ADOSC_Buy_Sell_Pressure
149: OBV_Buy_Sell_Pressure
150: BB_Overbought_Oversold_Signal
151: CCI_Overbought_Oversold_Signal
152: RSI_Overbought_Oversold_Signal
153: STOCH_Overbought_Oversold_Signal
154: STOCHF_Overbought_Oversold_Signal
155: STOCHRSI_Overbought_Oversold_Signal
156: ULTOSC_Overbought_Oversold_Signal
157: WILLR_Overbought_Oversold_Signal
158: BB_RSI_Reversal
159: BB_Volatility
160: ATR_Volatility
161: NATR_Volatility
162: TRANGE_Volatility
163: PATTERN_SUM
164: SENTIMENT_NEGATIVE
165: SENTIMENT_POSITIVE
166: SENTIMENT_UNCERTAINTY
167: SENTIMENT_LITIGIOUS
168: SENTIMENT_STRONG_MODAL
169: SENTIMENT_WEAK_MODAL
170: SENTIMENT_CONSTRAINING
171: NEXT_DAY_CLOSEPRICE
172: DAILY_CLOSEPRICE_CHANGE
173: DAILY_CLOSEPRICE_CHANGE_PERCENT
174: DAILY_CLOSEPRICE_DIRECTION
175: DAILY_MIDPRICE
176: NEXT_DAY_MIDPRICE
177: DAILY_MIDPRICE_CHANGE
178: DAILY_MIDPRICE_CHANGE_PERCENT
179: DAILY_MIDPRICE_DIRECTION
'''

def training_preprocess_data(df):
    # Handle missing and infinite values
    if df.isna().sum().sum() > 0 or df.isin([float('inf'), float('-inf')]).sum().sum() > 0:
        df = df.replace([float('inf'), float('-inf')], np.nan).dropna()

    # Create target variables before dropping columns
    y_classifier = (np.sign(df['DAILY_MIDPRICE_CHANGE']) >= 0).astype(int)
    y_scaler = StandardScaler()
    y_regressor = df[['DAILY_MIDPRICE_CHANGE']]  # Convert to DataFrame
    y_regressor_scaled = pd.DataFrame(y_scaler.fit_transform(y_regressor), columns=y_regressor.columns)

    # Copy the DataFrame for feature processing
    X = df.copy(deep=True)

    # Drop specific columns
    columns_to_drop = [
        'NEXT_DAY_CLOSEPRICE', 'DAILY_CLOSEPRICE_CHANGE', 'DAILY_CLOSEPRICE_CHANGE_PERCENT',
        'DAILY_CLOSEPRICE_DIRECTION',
        'DAILY_MIDPRICE', 'NEXT_DAY_MIDPRICE', 'DAILY_MIDPRICE_CHANGE', 'DAILY_MIDPRICE_CHANGE_PERCENT',
        'DAILY_MIDPRICE_DIRECTION',
        'Date',
    ]
    X = X.drop(columns=columns_to_drop)

    # Drop additional columns by index range if needed
    X = X.drop(X.columns[100:163], axis=1)

    # Standardize the features
    X_scaler = StandardScaler()
    X_scaled = pd.DataFrame(X_scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y_classifier, y_regressor_scaled

def predict_preprocess_data(df):
    # Select the last row and make a deep copy
    last_row_copy = df.iloc[-1].copy(deep=True)

    # Handle missing and infinite values
    if df.isna().sum().sum() > 0 or df.isin([float('inf'), float('-inf')]).sum().sum() > 0:
        df = df.replace([float('inf'), float('-inf')], np.nan).dropna()

    # Convert the last row copy to a DataFrame
    last_row_copy = pd.DataFrame([last_row_copy], columns=df.columns)

    # Concatenate the DataFrame with the last row copy
    df = pd.concat([df, last_row_copy], axis=0, ignore_index=True)

    # Scale the target variable
    y_scaler = StandardScaler()
    y_regressor = df[['DAILY_MIDPRICE_CHANGE']]  # Convert to DataFrame
    y_regressor_scaled = pd.DataFrame(y_scaler.fit_transform(y_regressor), columns=y_regressor.columns)

    # Make a deep copy of the DataFrame for features
    X = df.copy(deep=True)

    # Drop specific columns by name
    columns_to_drop = [
        'NEXT_DAY_CLOSEPRICE', 'DAILY_CLOSEPRICE_CHANGE', 'DAILY_CLOSEPRICE_CHANGE_PERCENT',
        'DAILY_CLOSEPRICE_DIRECTION',
        'DAILY_MIDPRICE', 'NEXT_DAY_MIDPRICE', 'DAILY_MIDPRICE_CHANGE', 'DAILY_MIDPRICE_CHANGE_PERCENT',
        'DAILY_MIDPRICE_DIRECTION',
        'Date',
    ]
    X = X.drop(columns=columns_to_drop)

    # Drop additional columns by index range if needed
    X = X.drop(X.columns[100:163], axis=1)

    # Scale the features
    X_scaler = StandardScaler()
    X_scaled = pd.DataFrame(X_scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y_scaler


