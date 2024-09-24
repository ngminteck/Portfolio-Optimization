import numpy as np
from sklearn.preprocessing import StandardScaler
from directory_manager import *

'''
0 : Date
1 : Open
2 : High
3 : Low
4 : Close
5 : Volume
6 : Change
7 : Wave
8 : EFP Volume
9 : EFS Volume
10 : Block Volume
11 : Last
12 : Previous Day Open Interest
13 : BB_UPPER
14 : BB_MIDDLE
15 : BB_LOWER
16 : DEMA
17 : EMA
18 : HT_TRENDLINE
19 : KAMA
20 : MA
21 : MAMA
22 : FAMA
23 : MAVP
24 : MIDPOINT
25 : MIDPRICE
26 : SAR
27 : SAREXT
28 : SMA
29 : T3
30 : TEMA
31 : TRIMA
32 : WMA
33 : APO
34 : CCI
35 : MACD
36 : MACD_signal
37 : MACD_hist
38 : MACDEXT
39 : MACDEXT_signal
40 : MACDEXT_hist
41 : MACDFIX
42 : MACDFIX_signal
43 : MACDFIX_hist
44 : MINUS_DM
45 : MOM
46 : PLUS_DM
47 : PPO
48 : ROC
49 : ROCP
50 : ROCR
51 : ROCR100
52 : TRIX
53 : AD
54 : ADOSC
55 : OBV
56 : ATR
57 : NATR
58 : TRANGE
59 : HT_DCPERIOD
60 : HT_DCPHASE
61 : HT_PHASOR_inphase
62 : HT_PHASOR_quadrature
63 : HT_SINE_sine
64 : HT_SINE_leadsine
65 : AVGPRICE
66 : MEDPRICE
67 : TYPPRICE
68 : WCLPRICE
69 : BETA
70 : LINEARREG
71 : LINEARREG_ANGLE
72 : LINEARREG_INTERCEPT
73 : LINEARREG_SLOPE
74 : STDDEV
75 : TSF
76 : VAR
77 : ADX
78 : ADXR
79 : AROON_down
80 : AROON_up
81 : AROONOSC
82 : CMO
83 : DX
84 : MFI
85 : MINUS_DI
86 : PLUS_DI
87 : RSI
88 : STOCH_slowk
89 : STOCH_slowd
90 : STOCHF_fastk
91 : STOCHF_fastd
92 : STOCHRSI_fastk
93 : STOCHRSI_fastd
94 : ULTOSC
95 : WILLR
96 : BOP
97 : HT_TRENDMODE
98 : CORREL
99 : DEMA_Trend
100 : EMA_Trend
101 : HT_TRENDLINE_Trend
102 : KAMA_Trend
103 : MA_Trend
104 : MAMA_Trend
105 : MAVP_Trend
106 : MIDPOINT_Trend
107 : MIDPRICE_Trend
108 : SAR_Trend
109 : SAREXT_Trend
110 : SMA_Trend
111 : T3_Trend
112 : TEMA_Trend
113 : TRIMA_Trend
114 : WMA_Trend
115 : ADX_Trend
116 : ADXR_Trend
117 : AROONOSC_Trend
118 : DX_Trend
119 : TRIX_Trend
120 : DMI_Trend
121 : AROON_Up_Trend
122 : AROON_Down_Trend
123 : PM_Uptrend
124 : PM_Downtrend
125 : ROC_Trend
126 : ROCP_Trend
127 : ROCR_Trend
128 : ROCR100_Trend
129 : ROC_Buy_Sell_Signal
130 : ROCP_Buy_Sell_Signal
131 : ROCR_Buy_Sell_Signal
132 : ROCR100_Buy_Sell_Signal
133 : APO_Buy_Sell_Signal
134 : MACD_Buy_Sell_Signal
135 : MACDEXT_Buy_Sell_Signal
136 : MACDFIX_Buy_Sell_Signal
137 : PPO_Buy_Sell_Signal
138 : MOM_Buy_Sell_Signal
139 : STOCH_Buy_Sell_Signal
140 : STOCHF_Buy_Sell_Signal
141 : STOCHRSI_Buy_Sell_Signal
142 : ULTOSC_Buy_Sell_Signal
143 : WILLR_Buy_Sell_Signal
144 : BOP_Buy_Sell_Pressure
145 : MFI_Buy_Sell_Pressure
146 : AD_Buy_Sell_Pressure
147 : ADOSC_Buy_Sell_Pressure
148 : OBV_Buy_Sell_Pressure
149 : BB_Overbought_Oversold_Signal
150 : CCI_Overbought_Oversold_Signal
151 : RSI_Overbought_Oversold_Signal
152 : STOCH_Overbought_Oversold_Signal
153 : STOCHF_Overbought_Oversold_Signal
154 : STOCHRSI_Overbought_Oversold_Signal
155 : ULTOSC_Overbought_Oversold_Signal
156 : WILLR_Overbought_Oversold_Signal
157 : BB_RSI_Reversal
158 : BB_Volatility
159 : ATR_Volatility
160 : NATR_Volatility
161 : TRANGE_Volatility
162 : PATTERN_SUM
163 : SENTIMENT_NEGATIVE
164 : SENTIMENT_POSITIVE
165 : SENTIMENT_UNCERTAINTY
166 : SENTIMENT_LITIGIOUS
167 : SENTIMENT_STRONG_MODAL
168 : SENTIMENT_WEAK_MODAL
169 : SENTIMENT_CONSTRAINING
170 : NEXT_DAY_CLOSEPRICE
171 : DAILY_CLOSEPRICE_CHANGE
172 : DAILY_CLOSEPRICE_CHANGE_PERCENT
173 : DAILY_CLOSEPRICE_DIRECTION
174 : DAILY_MIDPRICE
175 : NEXT_DAY_MIDPRICE
176 : DAILY_MIDPRICE_CHANGE
177 : DAILY_MIDPRICE_CHANGE_PERCENT
178 : DAILY_MIDPRICE_DIRECTION
'''

standard_scaled_columns = [
    "Open", "High", "Low", "Close", "Volume", "Change", "Wave", "EFP Volume", "EFS Volume",
    "Block Volume", "Last", "Previous Day Open Interest", "BB_UPPER", "BB_MIDDLE", "BB_LOWER",
    "DEMA", "EMA", "HT_TRENDLINE", "KAMA", "MA", "MAMA", "FAMA", "MAVP", "MIDPOINT", "MIDPRICE",
    "SAR", "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA", "APO", "CCI", "MACD", "MACD_signal",
    "MACD_hist", "MACDEXT", "MACDEXT_signal", "MACDEXT_hist", "MACDFIX", "MACDFIX_signal",
    "MACDFIX_hist", "MINUS_DM", "MOM", "PLUS_DM", "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
    "TRIX", "AD", "ADOSC", "OBV", "ATR", "NATR", "TRANGE", "HT_DCPERIOD", "HT_DCPHASE",
    "HT_PHASOR_inphase", "HT_PHASOR_quadrature", "HT_SINE_sine", "HT_SINE_leadsine", "AVGPRICE",
    "MEDPRICE", "TYPPRICE", "WCLPRICE", "BETA", "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT",
    "LINEARREG_SLOPE", "STDDEV", "TSF", "VAR"
]

division_hundred_columns = [
    "ADX", "ADXR", "AROON_down", "AROON_up", "AROONOSC", "CMO", "DX", "MFI", "MINUS_DI",
    "PLUS_DI", "RSI", "STOCH_slowk", "STOCH_slowd", "STOCHF_fastk", "STOCHF_fastd",
    "STOCHRSI_fastk", "STOCHRSI_fastd", "ULTOSC", "WILLR"
]

def training_preprocess_data(ticker_symbol):

    df = pd.read_csv(f"../data/all/{ticker_symbol}.csv")
    # Handle missing and infinite values
    if df.isna().sum().sum() > 0 or df.isin([float('inf'), float('-inf')]).sum().sum() > 0:
        df = df.replace([float('inf'), float('-inf')], np.nan).dropna()

    # Create target variables before dropping columns
    y_classifier = (np.sign(df['DAILY_CLOSEPRICE_CHANGE']) >= 0).astype(int)
    y_scaler = StandardScaler()
    y_regressor = df[['DAILY_CLOSEPRICE_CHANGE']]  # Convert to DataFrame
    y_regressor_scaled = pd.DataFrame(y_scaler.fit_transform(y_regressor), columns=y_regressor.columns)

    #for index, column in enumerate(df.columns):
     #   print(f"{index} : {column}")

    X = df.copy(deep=True)
    columns_to_drop = [
        'NEXT_DAY_CLOSEPRICE', 'DAILY_CLOSEPRICE_CHANGE', 'DAILY_CLOSEPRICE_CHANGE_PERCENT',
        'DAILY_CLOSEPRICE_DIRECTION',
        'DAILY_MIDPRICE', 'NEXT_DAY_MIDPRICE', 'DAILY_MIDPRICE_CHANGE', 'DAILY_MIDPRICE_CHANGE_PERCENT',
        'DAILY_MIDPRICE_DIRECTION',
        'Date',
    ]
    X = X.drop(columns=columns_to_drop)
    X = X.drop(X.columns[99:162], axis=1)

    # Drop columns with only one unique value
    X = X.loc[:, X.nunique() > 1]

    columns_to_divide_by_100 = [col for col in division_hundred_columns if col in X.columns]
    X[columns_to_divide_by_100] = X[columns_to_divide_by_100] / 100

    columns_to_standard_scale = [col for col in standard_scaled_columns if col in X.columns]
    X_scaler = StandardScaler()
    X[columns_to_standard_scale] = X_scaler.fit_transform(X[columns_to_standard_scale])

    file = f'{Trained_Feature_Folder}{ticker_symbol}.txt'
    with open(file, 'w') as f:
        for col in X.columns:
            f.write(col + '\n')

    print(X.shape)

    return X, y_classifier, y_regressor_scaled

def predict_preprocess_data(ticker_symbol):
    df = pd.read_csv(f"../data/all/{ticker_symbol}.csv")

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
    y_regressor = df[['DAILY_CLOSEPRICE_CHANGE']]  # Convert to DataFrame
    y_regressor_scaled = pd.DataFrame(y_scaler.fit_transform(y_regressor), columns=y_regressor.columns)


    file = f'{Trained_Feature_Folder}{ticker_symbol}.txt'
    with open(file, 'r') as f:
        column_names = f.read().splitlines()

    X = df.filter(column_names).copy()

    columns_to_divide_by_100 = [col for col in division_hundred_columns if col in X.columns]
    X[columns_to_divide_by_100] = X[columns_to_divide_by_100] / 100

    columns_to_standard_scale = [col for col in standard_scaled_columns if col in X.columns]
    X_scaler = StandardScaler()
    X[columns_to_standard_scale] = X_scaler.fit_transform(X[columns_to_standard_scale])

    print(X.shape)

    return X, y_scaler


