import numpy as np
import pandas as pd
import talib
# %%
def get_technical_indicator(df):

    # Bollinger Bands: Indicates overbought/oversold conditions
    df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # BB_UPPER: Upper Bollinger Band
    # BB_MIDDLE: Middle Bollinger Band (20-period moving average)
    # BB_LOWER: Lower Bollinger Band
    # Value range: [negative infinity, positive infinity]

    # Double Exponential Moving Average: Smooths price data
    df['DEMA'] = talib.DEMA(df['Close'], timeperiod=30)
    # DEMA: Double Exponential Moving Average with a period of 30
    # Value range: [negative infinity, positive infinity]

    # Exponential Moving Average: Smooths price data
    df['EMA'] = talib.EMA(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity

    # Hilbert Transform - Instantaneous Trendline: Identifies trend direction
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
    # Value range: [negative infinity, positive infinity]

    # Kaufman Adaptive Moving Average: Adjusts to market volatility
    df['KAMA'] = talib.KAMA(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity]

    # Moving Average: Smooths price data
    df['MA'] = talib.MA(df['Close'], timeperiod=30, matype=0)
    # Value range: [negative infinity, positive infinity]

    # MESA Adaptive Moving Average: Adapts to market cycles
    df['MAMA'], df['FAMA'] = talib.MAMA(df['Close'], fastlimit=0.5, slowlimit=0.05)
    # Value range: [negative infinity, positive infinity

    # Moving Average with Variable Period: Smooths price data with variable periods
    df['MAVP'] = talib.MAVP(df['Close'], df['Volume'], minperiod=2, maxperiod=30, matype=0)
    # Value range: [negative infinity, positive infinity]

    # MidPoint over Period: Average of the highest and lowest prices
    df['MIDPOINT'] = talib.MIDPOINT(df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]

    # Midpoint Price over Period: Average of the highest and lowest prices
    df['MIDPRICE'] = talib.MIDPRICE(df['High'], df['Low'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]

    # Parabolic SAR: Identifies potential reversal points
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    # Value range: [negative infinity, positive infinity]

    # Parabolic SAR - Extended: Identifies potential reversal points with extended parameters
    df['SAREXT'] = talib.SAREXT(df['High'], df['Low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    # Value range: [negative infinity, positive infinity]

    # Simple Moving Average: Smooths price data
    df['SMA'] = talib.SMA(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity]

    # Triple Exponential Moving Average (T3): Smooths price data with less lag
    df['T3'] = talib.T3(df['Close'], timeperiod=5, vfactor=0.7)
    # Value range: [negative infinity, positive infinity]

    # Triple Exponential Moving Average: Smooths price data
    df['TEMA'] = talib.TEMA(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity]

    # Triangular Moving Average: Smooths price data
    df['TRIMA'] = talib.TRIMA(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity]

    # Weighted Moving Average: Smooths price data
    df['WMA'] = talib.WMA(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity]

    # Momentum Indicators


    # Absolute Price Oscillator: Measures momentum
    df['APO'] = talib.APO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive APO values indicate upward momentum. Negative APO values indicate downward momentum.


    # Commodity Channel Index: Identifies cyclical trends
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: CCI values above 100 indicate overbought conditions. CCI values below -100 indicate oversold conditions.


    # Moving Average Convergence/Divergence: Measures momentum
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive MACD values indicate upward momentum. Negative MACD values indicate downward momentum.

    # MACD with controllable MA type: Measures momentum
    df['MACDEXT'], df['MACDEXT_signal'], df['MACDEXT_hist'] = talib.MACDEXT(df['Close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive MACDEXT values indicate upward momentum. Negative MACDEXT values indicate downward momentum.

    # MACD Fix 12/26: Measures momentum
    df['MACDFIX'], df['MACDFIX_signal'], df['MACDFIX_hist'] = talib.MACDFIX(df['Close'], signalperiod=9)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive MACDFIX values indicate upward momentum. Negative MACDFIX values indicate downward momentum.

    # Minus Directional Movement: Measures trend strength
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive MINUS_DM values indicate downward movement.

    # Momentum: Measures momentum
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive MOM values indicate upward momentum. Negative MOM values indicate downward momentum.

    # Plus Directional Movement: Measures trend strength
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive PLUS_DM values indicate upward movement.

    # Percentage Price Oscillator: Measures momentum
    df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive PPO values indicate upward momentum. Negative PPO values indicate downward momentum.

    # Rate of Change: Measures rate of change
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive ROC values indicate upward momentum. Negative ROC values indicate downward momentum.

    # Rate of Change Percentage: Measures rate of change percentage
    df['ROCP'] = talib.ROCP(df['Close'], timeperiod=10)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive ROCP values indicate upward momentum. Negative ROCP values indicate downward momentum.

    # Rate of Change Ratio: Measures rate of change ratio
    df['ROCR'] = talib.ROCR(df['Close'], timeperiod=10)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive ROCR values indicate upward momentum. Negative ROCR values indicate downward momentum.

    # Rate of Change Ratio 100 Scale: Measures rate of change ratio scaled by 100
    df['ROCR100'] = talib.ROCR100(df['Close'], timeperiod=10)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive ROCR100 values indicate upward momentum. Negative ROCR100 values indicate downward momentum.

    # TRIX: Measures rate of change of a triple smoothed EMA
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive TRIX values indicate upward momentum. Negative TRIX values indicate downward momentum.



    # Volume Indicators

    # Chaikin A/D Line: Measures accumulation/distribution
    df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    # Value range: [negative infinity, positive infinity]
    # Example: Positive AD values indicate accumulation (buying pressure). Negative AD values indicate distribution (selling pressure).

    # Chaikin A/D Oscillator: Measures momentum of the A/D line
    df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    # Value range: [negative infinity, positive infinity]
    # Example: Positive ADOSC values indicate upward momentum. Negative ADOSC values indicate downward momentum.

    # On Balance Volume: Measures buying and selling pressure
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    # Value range: [negative infinity, positive infinity]
    # Example: Positive OBV values indicate buying pressure. Negative OBV values indicate selling pressure.

    # Volatility Indicators

    # Average True Range: Measures volatility
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: Higher ATR values indicate higher volatility.

    # Normalized Average True Range: Measures normalized volatility
    df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: Higher NATR values indicate higher normalized volatility.

    # True Range: Measures true range
    df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: Higher TRANGE values indicate a larger range between high, low, and close prices.

    # Cycle Indicators

    # Hilbert Transform - Dominant Cycle Period: Identifies dominant cycle period
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: Higher HT_DCPERIOD values indicate a longer dominant cycle period.

    # Hilbert Transform - Dominant Cycle Phase: Identifies dominant cycle phase
    df['HT_DCPHASE'] = talib.HT_DCPHASE(df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: HT_DCPHASE values oscillate between 0 and 360 degrees, indicating the phase of the dominant cycle.

    # Hilbert Transform - Phasor Components: Identifies phasor components
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: Inphase and quadrature components help identify the position within the cycle.

    # Hilbert Transform - SineWave: Identifies sinewave components
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: Sine and leadsine values help identify turning points in the cycle.


    # Price Transform

    # Average Price: Calculates average price
    df['AVGPRICE'] = talib.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: The average price is calculated as (Open + High + Low + Close) / 4.

    # Median Price: Calculates median price
    df['MEDPRICE'] = talib.MEDPRICE(df['High'], df['Low'])
    # Value range: [negative infinity, positive infinity]
    # Example: The median price is calculated as (High + Low) / 2.

    # Typical Price: Calculates typical price
    df['TYPPRICE'] = talib.TYPPRICE(df['High'], df['Low'], df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: The typical price is calculated as (High + Low + Close) / 3.

    # Weighted Close Price: Calculates weighted close price
    df['WCLPRICE'] = talib.WCLPRICE(df['High'], df['Low'], df['Close'])
    # Value range: [negative infinity, positive infinity]
    # Example: The weighted close price is calculated as (High + Low + 2 * Close) / 4.

    # Statistic Functions

    # Beta: Measures volatility relative to the market
    df['BETA'] = talib.BETA(df['High'], df['Low'], timeperiod=5)
    # Value range: [negative infinity, positive infinity]
    # Example: A BETA value greater than 1 indicates higher volatility relative to the market. A BETA value less than 1 indicates lower volatility.


    # Linear Regression: Calculates linear regression
    df['LINEARREG'] = talib.LINEARREG(df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: The LINEARREG value represents the predicted close price based on the linear regression model.

    # Linear Regression Angle: Calculates linear regression angle
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: A positive LINEARREG_ANGLE indicates an upward trend. A negative LINEARREG_ANGLE indicates a downward trend.

    # Linear Regression Intercept: Calculates linear regression intercept
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: The LINEARREG_INTERCEPT value represents the intercept of the linear regression line with the y-axis.

    # Linear Regression Slope: Calculates linear regression slope
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: A positive LINEARREG_SLOPE indicates an upward trend. A negative LINEARREG_SLOPE indicates a downward trend.

    # Standard Deviation: Measures volatility
    df['STDDEV'] = talib.STDDEV(df['Close'], timeperiod=5, nbdev=1)
    # Value range: [0, positive infinity]
    # Example: Higher STDDEV values indicate higher volatility.

    # Time Series Forecast: Forecasts future values
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    # Value range: [negative infinity, positive infinity]
    # Example: The TSF value represents the forecasted close price based on the time series model.

    # Variance: Measures volatility
    df['VAR'] = talib.VAR(df['Close'], timeperiod=5, nbdev=1)
    # Value range: [0, positive infinity]
    # Example: Higher VAR values indicate higher volatility.

    # 100 range
    # Average Directional Movement Index: Measures trend strength
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [0, 100]
    # Example: Higher ADX values indicate a stronger trend. Values above 25 suggest a strong trend.

    # Average Directional Movement Index Rating: Measures trend strength
    df['ADXR'] = talib.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [0, 100]
    # Example: Higher ADXR values indicate a stronger trend. Values above 25 suggest a strong trend.

    # Aroon: Identifies trend changes
    df['AROON_down'], df['AROON_up'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
    # Value range: [0, 100]
    # Example: Higher AROON_up values indicate a stronger uptrend. Higher AROON_down values indicate a stronger downtrend.

    # Aroon Oscillator: Measures trend strength
    df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
    # Value range: [-100, 100]
    # Example: Positive AROONOSC values indicate upward momentum. Negative AROONOSC values indicate downward momentum.

    # Chande Momentum Oscillator: Measures momentum
    df['CMO'] = talib.CMO(df['Close'], timeperiod=14)
    # Value range: [-100, 100]
    # Example: Positive CMO values indicate upward momentum. Negative CMO values indicate downward momentum.

    # Directional Movement Index: Measures trend strength
    df['DX'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [0, 100]
    # Example: Higher DX values indicate a stronger trend. Values above 25 suggest a strong trend.

    # Money Flow Index: Measures buying and selling pressure
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    # Value range: [0, 100]
    # Example: MFI values above 80 indicate overbought conditions. MFI values below 20 indicate oversold conditions.

    # Minus Directional Indicator: Measures trend strength
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [0, 100]
    # Example: Higher MINUS_DI values indicate a stronger downtrend.

    # Plus Directional Indicator: Measures trend strength
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [0, 100]
    # Example: Higher PLUS_DI values indicate a stronger uptrend.

    # Relative Strength Index: Measures momentum
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    # Value range: [0, 100]
    # Example: RSI values above 70 indicate overbought conditions (potential fall). RSI values below 30 indicate oversold conditions (potential rise).

    # Stochastic: Measures momentum
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # Value range: [0, 100]
    # Example: Stochastic values above 80 indicate overbought conditions (potential fall). Stochastic values below 20 indicate oversold conditions (potential rise).

    # Stochastic Fast: Measures momentum
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
    # Value range: [0, 100]
    # Example: Fast Stochastic values above 80 indicate overbought conditions (potential fall). Fast Stochastic values below 20 indicate oversold conditions (potential rise).

    # Stochastic Relative Strength Index: Measures momentum
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0)
    # Value range: [0, 100]
    # Example: Stochastic RSI values above 80 indicate overbought conditions (potential fall). Stochastic RSI values below 20 indicate oversold conditions (potential rise).

    # Ultimate Oscillator: Measures momentum
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # Value range: [0, 100]
    # Example: ULTOSC values above 70 indicate overbought conditions (potential fall). ULTOSC values below 30 indicate oversold conditions (potential rise).

    # Williams' %R: Measures overbought/oversold conditions
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Value range: [-100, 0]
    # Example: WILLR values above -20 indicate overbought conditions (potential fall). WILLR values below -80 indicate oversold conditions (potential rise).

    # Balance Of Power: Measures buying and selling pressure
    df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
    # Value range: [-1, 1]
    # Example: Positive BOP values indicate buying pressure. Negative BOP values indicate selling pressure.

    # Hilbert Transform - Trend vs Cycle Mode: Identifies trend vs cycle mode
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['Close'])
    # Value range: [0, 1]
    # Example: A value of 1 indicates a trending market. A value of 0 indicates a cyclical market.

    # Pearson's Correlation Coefficient (r): Measures correlation
    df['CORREL'] = talib.CORREL(df['High'], df['Low'], timeperiod=30)
    # Value range: [-1, 1]
    # Example: A CORREL value close to 1 indicates a strong positive correlation. A CORREL value close to -1 indicates a strong negative correlation.


    # Create new columns in a separate DataFrame
    new_columns = pd.DataFrame(index=df.index)

    # Trend, 1 indicate rise, -1 indicate fall and 0 indicate neutral
    new_columns['DEMA_Trend'] = np.where(df['Close'] > df['DEMA'], 1, np.where(df['Close'] < df['DEMA'], -1, 0))
    new_columns['EMA_Trend'] = np.where(df['Close'] > df['EMA'], 1, np.where(df['Close'] < df['EMA'], -1, 0))
    new_columns['HT_TRENDLINE_Trend'] = np.where(df['Close'] > df['HT_TRENDLINE'], 1, np.where(df['Close'] < df['HT_TRENDLINE'], -1, 0))
    new_columns['KAMA_Trend'] = np.where(df['Close'] > df['KAMA'], 1, np.where(df['Close'] < df['KAMA'], -1, 0))
    new_columns['MA_Trend'] = np.where(df['Close'] > df['MA'], 1, np.where(df['Close'] < df['MA'], -1, 0))
    new_columns['MAMA_Trend'] = np.where(df['MAMA'] > df['FAMA'], 1, np.where(df['MAMA'] < df['FAMA'], -1, 0))
    new_columns['MAVP_Trend'] = np.where(df['Close'] > df['MAVP'], 1, np.where(df['Close'] < df['MAVP'], -1, 0))
    new_columns['MIDPOINT_Trend'] = np.where(df['Close'] > df['MIDPOINT'], 1, np.where(df['Close'] < df['MIDPOINT'], -1, 0))
    new_columns['MIDPRICE_Trend'] = np.where(df['Close'] > df['MIDPRICE'], 1, np.where(df['Close'] < df['MIDPRICE'], -1, 0))
    new_columns['SAR_Trend'] = np.where(df['Close'] > df['SAR'], 1, np.where(df['Close'] < df['SAR'], -1, 0))
    new_columns['SAREXT_Trend'] = np.where(df['Close'] > df['SAREXT'], 1, np.where(df['Close'] < df['SAREXT'], -1, 0))
    new_columns['SMA_Trend'] = np.where(df['Close'] > df['SMA'], 1, np.where(df['Close'] < df['SMA'], -1, 0))
    new_columns['T3_Trend'] = np.where(df['Close'] > df['T3'], 1, np.where(df['Close'] < df['T3'], -1, 0))
    new_columns['TEMA_Trend'] = np.where(df['Close'] > df['TEMA'], 1, np.where(df['Close'] < df['TEMA'], -1, 0))
    new_columns['TRIMA_Trend'] = np.where(df['Close'] > df['TRIMA'], 1, np.where(df['Close'] < df['TRIMA'], -1, 0))
    new_columns['WMA_Trend'] = np.where(df['Close'] > df['WMA'], 1, np.where(df['Close'] < df['WMA'], -1, 0))
    new_columns['ADX_Trend'] = np.where(df['ADX'] > 25, 1, np.where(df['ADX'] <= 25, -1, 0))
    new_columns['ADXR_Trend'] = np.where(df['ADXR'] > 25, 1, np.where(df['ADXR'] <= 25, -1, 0))
    new_columns['AROONOSC_Trend'] = np.where(df['AROONOSC'] > 0, 1, np.where(df['AROONOSC'] <= 0, -1, 0))
    new_columns['DX_Trend'] = np.where(df['DX'] > 25, 1, np.where(df['DX'] <= 25, -1, 0))
    new_columns['TRIX_Trend'] = np.where(df['TRIX'] > 0, 1, np.where(df['TRIX'] <= 0, -1, 0))
    new_columns['DMI_Trend'] = np.where(df['PLUS_DI'] > df['MINUS_DI'], 1, np.where(df['PLUS_DI'] <= df['MINUS_DI'], -1, 0))

    new_columns['AROON_Up_Trend'] = np.where(df['AROON_up'] > 50, 1, np.where(df['AROON_up'] <= 50, -1, 0))
    new_columns['AROON_Down_Trend'] = np.where(df['AROON_down'] > 50, 1, np.where(df['AROON_down'] <= 50, -1, 0))
    new_columns['PM_Uptrend'] = np.where(df['PLUS_DI'] > df['MINUS_DI'], 1, np.where(df['PLUS_DI'] <= df['MINUS_DI'], -1, 0))
    new_columns['PM_Downtrend'] = np.where(df['MINUS_DI'] > df['PLUS_DI'], 1, np.where(df['MINUS_DI'] <= df['PLUS_DI'], -1, 0))

    new_columns['ROC_Trend'] = np.where(df['ROC'] > 0, 1, np.where(df['ROC'] <= 0, -1, 0))
    new_columns['ROCP_Trend'] = np.where(df['ROCP'] > 0, 1, np.where(df['ROCP'] <= 0, -1, 0))
    new_columns['ROCR_Trend'] = np.where(df['ROCR'] > 0, 1, np.where(df['ROCR'] <= 0, -1, 0))
    new_columns['ROCR100_Trend'] = np.where(df['ROCR100'] > 0, 1, np.where(df['ROCR100'] <= 0, -1, 0))

    # Buy/Sell Signal, 1 indicate buy which will rise, -1 indicate sell which will fall and 0 indicate neutral

    new_columns['ROC_Buy_Sell_Signal'] = np.where(df['ROC'] > 0, 1, np.where(df['ROC'] < 0, -1, 0))
    new_columns['ROCP_Buy_Sell_Signal'] = np.where(df['ROCP'] > 0, 1, np.where(df['ROCP'] < 0, -1, 0))
    new_columns['ROCR_Buy_Sell_Signal'] = np.where(df['ROCR'] > 0, 1, np.where(df['ROCR'] < 0, -1, 0))
    new_columns['ROCR100_Buy_Sell_Signal'] = np.where(df['ROCR100'] > 0, 1, np.where(df['ROCR100'] < 0, -1, 0))

    new_columns['APO_Buy_Sell_Signal'] = np.where(df['APO'] > 0, 1, -1)
    new_columns['MACD_Buy_Sell_Signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
    new_columns['MACDEXT_Buy_Sell_Signal'] = np.where(df['MACDEXT'] > df['MACDEXT_signal'], 1, -1)
    new_columns['MACDFIX_Buy_Sell_Signal'] = np.where(df['MACDFIX'] > df['MACDFIX_signal'], 1, -1)
    new_columns['PPO_Buy_Sell_Signal'] = np.where(df['PPO'] > 0, 1, -1)
    new_columns['MOM_Buy_Sell_Signal'] = np.where(df['MOM'] > 0, 1, -1)
    new_columns['STOCH_Buy_Sell_Signal'] = np.where(df['STOCH_slowk'] > df['STOCH_slowd'], 1, -1)
    new_columns['STOCHF_Buy_Sell_Signal'] = np.where(df['STOCHF_fastk'] > df['STOCHF_fastd'], 1, -1)
    new_columns['STOCHRSI_Buy_Sell_Signal'] = np.where(df['STOCHRSI_fastk'] > df['STOCHRSI_fastd'], 1, -1)
    new_columns['ULTOSC_Buy_Sell_Signal'] = np.where(df['ULTOSC'] > 50, 1, -1)
    new_columns['WILLR_Buy_Sell_Signal'] = np.where(df['WILLR'] > -80, 1, np.where(df['WILLR'] < -20, -1, 0))

    # Buy/Sell Pressure, 1 indicate buy pressure which will rise, -1 indicate sell pressure which will fall and 0 indicate neutral
    new_columns['BOP_Buy_Sell_Pressure'] = np.where(df['BOP'] <= 0, -1, 1)
    new_columns['MFI_Buy_Sell_Pressure'] = np.where(df['MFI'] <= 50, -1, 1)
    new_columns['AD_Buy_Sell_Pressure'] = np.where(df['AD'] > 0, 1, -1)
    new_columns['ADOSC_Buy_Sell_Pressure'] = np.where(df['ADOSC'] > 0, 1, -1)
    new_columns['OBV_Buy_Sell_Pressure'] = np.where(df['OBV'] > 0, 1, -1)

    # Overbought/Oversold Indicators,1 indicate overbought which will rise, -1 indicate oversold which will fall and 0 indicate neutral
    new_columns['BB_Overbought_Oversold_Signal'] = np.where(df['Close'] <= df['BB_LOWER'], -1, np.where(df['Close'] >= df['BB_UPPER'], 1, 0))
    new_columns['CCI_Overbought_Oversold_Signal'] = np.where(df['CCI'] < -100, -1, np.where(df['CCI'] > 100, 1, 0))
    new_columns['RSI_Overbought_Oversold_Signal'] = np.where(df['RSI'] < 30, -1, np.where(df['RSI'] > 70, 1, 0))
    new_columns['STOCH_Overbought_Oversold_Signal'] = np.where(df['STOCH_slowk'] > 80, 1, np.where(df['STOCH_slowk'] < 20, -1, 0))
    new_columns['STOCHF_Overbought_Oversold_Signal'] = np.where(df['STOCHF_fastk'] > 80, 1, np.where(df['STOCHF_fastk'] < 20, -1, 0))
    new_columns['STOCHRSI_Overbought_Oversold_Signal'] = np.where(df['STOCHRSI_fastk'] > 80, 1, np.where(df['STOCHRSI_fastk'] < 20, -1, 0))
    new_columns['ULTOSC_Overbought_Oversold_Signal'] = np.where(df['ULTOSC'] > 70, 1, np.where(df['ULTOSC'] < 30, -1, 0))
    new_columns['WILLR_Overbought_Oversold_Signal'] = np.where(df['WILLR'] > -20, 1, np.where(df['WILLR'] < -80, -1, 0))

    ## Reserve Indicators, 1 indicate going bullish which will rise, -1 indicate going bearish which will fall and 0 indicate neutral
    new_columns['BB_RSI_Reversal'] = np.where \
        ((df['Close'] < df['BB_LOWER']) & (df['RSI'] < 30) & (df['RSI'].shift(1) < df['RSI']), 1, np.where((df['Close'] > df['BB_UPPER']) & (df['RSI'] > 70) & (df['RSI'].shift(1) > df['RSI']), -1, 0))

    ## Volatility Indicators, 1 indicate low volatility, -1 indicate high volatitlity and 0 indicate neutral
    new_columns['BB_Volatility'] = np.where((df['Close'] > df['BB_UPPER']) | (df['Close'] < df['BB_LOWER']), -1, np.where((df['Close'] <= df['BB_UPPER']) & (df['Close'] >= df['BB_LOWER']), 1, 0))
    new_columns['ATR_Volatility'] = np.where(df['ATR'] > df['ATR'].rolling(window=14).mean(), -1, np.where(df['ATR'] <= df['ATR'].rolling(window=14).mean(), 1, 0))
    new_columns['NATR_Volatility'] = np.where(df['NATR'] > df['NATR'].rolling(window=14).mean(), -1, np.where(df['NATR'] <= df['NATR'].rolling(window=14).mean(), 1, 0))
    new_columns['TRANGE_Volatility'] = np.where(df['TRANGE'] > df['TRANGE'].rolling(window=14).mean(), -1, np.where(df['TRANGE'] <= df['TRANGE'].rolling(window=14).mean(), 1, 0))


    # Pattern Recognition

    # Get all candlestick pattern functions
    all_functions = talib.get_function_groups()
    candlestick_patterns = all_functions['Pattern Recognition']
    patterns = {pattern: getattr(talib, pattern) for pattern in candlestick_patterns}

    # Initialize Pattern_Sum column
    new_columns['PATTERN_SUM'] = 0

    # Apply each pattern function to the DataFrame and sum the results
    for pattern_name, pattern_func in patterns.items():
        pattern_result = pattern_func(df['Open'], df['High'], df['Low'], df['Close'])
        new_columns['PATTERN_SUM'] += pattern_result

    # Normalize the summed pattern values to be within the range of -1 to 1
    new_columns['PATTERN_SUM'] = new_columns['PATTERN_SUM'].apply(lambda x: np.clip(x, -100, 100) / 100)
    # Value range: [-1, 1]
    # Example: -1 bearish , 0 no detection, 1 bullish

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, new_columns], axis=1)

    return df