import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import pandas as pd
from scipy import signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import mysql.connector
from IPython.display import display
import yfinance as yf



def extrema(symbol, time_frame, strategy):
    if time_frame == 'M1':
        timeframe = mt5.TIMEFRAME_M1
        interval = 525600

    if time_frame == 'M5':
        timeframe = mt5.TIMEFRAME_M5
        interval = 105120

    if time_frame == 'M15':
        timeframe = mt5.TIMEFRAME_M15
        interval = 35040

    if time_frame == 'M30':
        timeframe = mt5.TIMEFRAME_M30
        interval = 17520

    if time_frame == 'H1':
        timeframe = mt5.TIMEFRAME_H1
        interval = 8760

    if time_frame == 'H4':
        timeframe = mt5.TIMEFRAME_H4
        interval = 2190

    if time_frame == 'D1':
        timeframe = mt5.TIMEFRAME_D1
        interval = 365

    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, interval)
    data = pd.DataFrame(data)
    # print(data)

    # Rename columns
    new_column_names = {'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'time': 'Date'}

    data.rename(columns=new_column_names, inplace=True)
    data['Date'] = np.array(pd.to_datetime(data['Date'], unit='s'))
    data.set_index('Date', inplace=True)
    end_date = pd.Timestamp('2023-11-18 00:00:00')

    if time_frame != 'M5' or time_frame != 'M1':
        start_date = end_date - timedelta(days=365)

    if time_frame == 'M5' or time_frame == 'M1':
        start_date = end_date - timedelta(days=90)

    data = data[(start_date <= data.index) & (data.index <= end_date)]



    # start_date = pd.Timestamp('2022-11-18 00:00:00')
    # start_date = int(start_date.to_pydatetime().timestamp())

    # end_date = pd.Timestamp('2023-11-18 00:00:00')
    # end_date = int(end_date.to_pydatetime().timestamp())

    # data = yf.download('AVAX-USD', start=start_date, end=end_date, interval='1d', progress=False)



    c = data['Close']
    o = data['Open']

    h = data['High']
    l = data['Low']

    MA_h = h.rolling(window=14).mean()
    MA_l = l.rolling(window=14).mean()

    for i in range(len(h)-1, 0, -1):
        if h.iloc[i] == h.iloc[i-1]:
            h = h.drop(h.index[i-1])

    for i in range(len(l)-1, 0, -1):
        if l.iloc[i] == l.iloc[i-1]:
            l = l.drop(l.index[i-1])



    #EXTREMA 1
    highs_peaks = signal.argrelextrema(h.values, np.greater)[0]
    lows_valleys = signal.argrelextrema(l.values, np.less)[0]

    if highs_peaks[0] == lows_valleys[0]:
        highs_peaks = highs_peaks[1:]
        lows_valleys = lows_valleys[1:]

    # High and Low Merge peaks and valleys data points using pandas extrema
    ex1_df_highs_peaks = pd.DataFrame({'Date': h.index[highs_peaks], 'Extremum': h.values[highs_peaks]})
    ex1_df_lows_valleys = pd.DataFrame({'Date': l.index[lows_valleys], 'Extremum': l.values[lows_valleys]})

    ex1_df_highs_peaks_and_lows_valleys = pd.concat([ex1_df_highs_peaks, ex1_df_lows_valleys], axis=0, ignore_index=True, sort=True)
    ex1_df_highs_peaks_and_lows_valleys = ex1_df_highs_peaks_and_lows_valleys.sort_values(by=['Date'])

    # Apply the ZigZag filter
    ex1_extrema = []
    ex1_actual_highs = []
    ex1_actual_lows = []
    ex1_extrema.append((ex1_df_highs_peaks_and_lows_valleys.iat[0, 0], ex1_df_highs_peaks_and_lows_valleys.iat[0, 1]))

    if ex1_extrema[0][0] == ex1_df_highs_peaks.iat[0, 0]:
        ex1_actual_highs.append((ex1_df_highs_peaks.iat[0, 0], ex1_df_highs_peaks.iat[0, 1]))

    if ex1_extrema[0][0] == ex1_df_lows_valleys.iat[0, 0]:
        ex1_actual_lows.append((ex1_df_lows_valleys.iat[0, 0], ex1_df_lows_valleys.iat[0, 1]))

    ex1_df_highs_peaks_and_lows_valleys = ex1_df_highs_peaks_and_lows_valleys[ex1_df_highs_peaks_and_lows_valleys['Date'] > ex1_extrema[-1][0]]

    for i in range(len(ex1_df_highs_peaks_and_lows_valleys)):
        # Missing if the Date is the Same
        if (ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'] == ex1_extrema[-1][0]):
            continue

        # If there are 2 EXTREMUM on the Same Date
        if (((ex1_df_highs_peaks_and_lows_valleys.iloc[i][['Date']] == ex1_df_highs_peaks[['Date']]).all(axis=1).any()) and 
        ((ex1_df_highs_peaks_and_lows_valleys.iloc[i][['Date']] == ex1_df_lows_valleys[['Date']]).all(axis=1).any()) and
        (ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'] > ex1_extrema[-1][0])):
            extrema_common_date = ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date']

            # Add Extrema High if previous is Low
            if ex1_actual_lows:
                if (ex1_extrema[-1][0] == ex1_actual_lows[-1][0]):
                    extremum_value = ex1_df_highs_peaks[ex1_df_highs_peaks['Date'] == extrema_common_date]['Extremum'].values[0]
                    ex1_actual_highs.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], extremum_value))
                    ex1_extrema.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], extremum_value))
                    continue

            # Add Extrema Low if previous is High
            if ex1_actual_highs:
                if (ex1_extrema[-1][0] == ex1_actual_highs[-1][0]):
                    extremum_value = ex1_df_lows_valleys[ex1_df_lows_valleys['Date'] == extrema_common_date]['Extremum'].values[0]
                    ex1_actual_lows.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], extremum_value))
                    ex1_extrema.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], extremum_value))
                    continue

        # Add Extrema High if previous is Low
        if ex1_actual_lows:
            if (((ex1_df_highs_peaks_and_lows_valleys.iloc[i][['Date']] == ex1_df_highs_peaks[['Date']]).all(axis=1).any()) and (ex1_extrema[-1][0] == ex1_actual_lows[-1][0]) and (ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'] > ex1_extrema[-1][0])):
                ex1_actual_highs.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))
                ex1_extrema.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))

        # Add Extrema Low if previous is High
        if ex1_actual_highs:
            if (((ex1_df_highs_peaks_and_lows_valleys.iloc[i][['Date']] == ex1_df_lows_valleys[['Date']]).all(axis=1).any()) and (ex1_extrema[-1][0] == ex1_actual_highs[-1][0]) and (ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'] > ex1_extrema[-1][0])):
                ex1_actual_lows.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))
                ex1_extrema.append((ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex1_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))


    # Convert ex1_extrema to DataFrame
    ex1_df_extrema = pd.DataFrame(ex1_extrema, columns=['Date', 'Extremum'])
    ex1_df_extrema = ex1_df_extrema.sort_values(by='Date')



    #EXTREMA 2
    # Extrema Points from EXTREMA 1
    df_list_peaks_dates = ex1_df_highs_peaks['Date'].tolist()
    df_list_peaks_extrema = ex1_df_highs_peaks['Extremum'].tolist()

    df_list_valleys_dates = ex1_df_lows_valleys['Date'].tolist()
    df_list_valleys_extrema = ex1_df_lows_valleys['Extremum'].tolist()

    # Create a new DataFrame to store the new data
    ex2_df_highs_peaks = []
    ex2_df_lows_valleys = []

    # Iterate through ex1_extrema and add new data to the ex2_df_highs_peaks list
    for date, extremum in ex1_extrema:
        if date in df_list_peaks_dates:
            if extremum == df_list_peaks_extrema[df_list_peaks_dates.index(date)]:
                ex2_df_highs_peaks.append({'Date': date, 'Extremum': extremum})


    for date, extremum in ex1_extrema:
        if date in df_list_valleys_dates:
            if extremum == df_list_valleys_extrema[df_list_valleys_dates.index(date)]:
                ex2_df_lows_valleys.append({'Date': date, 'Extremum': extremum})


    # Convert the ex2_df_highs_peaks list to a DataFrame
    ex2_df_highs_peaks = pd.DataFrame(ex2_df_highs_peaks)
    ex2_df_lows_valleys = pd.DataFrame(ex2_df_lows_valleys)

    for i in range(len(ex2_df_highs_peaks)-1, 0, -1):
        if ex2_df_highs_peaks.iat[i, 1] == ex2_df_highs_peaks.iat[i-1, 1]:
            ex2_df_highs_peaks = ex2_df_highs_peaks.drop(ex2_df_highs_peaks.index[i-1])


    for i in range(len(ex2_df_lows_valleys)-1, 0, -1):
        if ex2_df_lows_valleys.iat[i, 1] == ex2_df_lows_valleys.iat[i-1, 1]:
            ex2_df_lows_valleys = ex2_df_lows_valleys.drop(ex2_df_lows_valleys.index[i-1])


    ex2_df_highs_peaks = ex2_df_highs_peaks.reset_index(drop=True)
    ex2_df_lows_valleys = ex2_df_lows_valleys.reset_index(drop=True)

    # Find local extrema using argrelextrema
    ex2_extrema_highs_peaks = signal.argrelextrema(ex2_df_highs_peaks['Extremum'].values, np.greater, order=1)
    ex2_extrema_lows_valleys = signal.argrelextrema(ex2_df_lows_valleys['Extremum'].values, np.less, order=1)

    # Highs Peaks
    ex2_df_highs_peaks = pd.DataFrame({
        'Date': ex2_df_highs_peaks.loc[ex2_extrema_highs_peaks[0], 'Date'],
        'Extremum': ex2_df_highs_peaks.loc[ex2_extrema_highs_peaks[0], 'Extremum']
    })

    # Lows Valleys
    ex2_df_lows_valleys = pd.DataFrame({
        'Date': ex2_df_lows_valleys.loc[ex2_extrema_lows_valleys[0], 'Date'],
        'Extremum': ex2_df_lows_valleys.loc[ex2_extrema_lows_valleys[0], 'Extremum']
    })

    # Concatenate the two dataframes
    ex2_df_highs_peaks_and_lows_valleys = pd.concat([ex2_df_highs_peaks, ex2_df_lows_valleys], ignore_index=True)
    ex2_df_highs_peaks_and_lows_valleys = pd.DataFrame(ex2_df_highs_peaks_and_lows_valleys, columns=['Date', 'Extremum'])
    ex2_df_highs_peaks_and_lows_valleys = ex2_df_highs_peaks_and_lows_valleys.sort_values(by='Date')

    ex2_extrema = []
    ex2_df_extrema = []
    ex2_actual_highs = []
    ex2_actual_lows = []
    ex2_extrema.append((ex2_df_highs_peaks_and_lows_valleys['Date'].iloc[0], ex2_df_highs_peaks_and_lows_valleys['Extremum'].iloc[0]))

    if ex2_extrema[0][0] == ex2_df_highs_peaks.iat[0, 0]:
        ex2_actual_highs.append((ex2_df_highs_peaks.iat[0, 0], ex2_df_highs_peaks.iat[0, 1]))

        ex2_actual_lows.append((ex2_df_lows_valleys.iat[0, 0], ex2_df_lows_valleys.iat[0, 1]))
        ex2_extrema.append((ex2_df_lows_valleys.iat[0, 0], ex2_df_lows_valleys.iat[0, 1]))

    if ex2_extrema[0][0] == ex2_df_lows_valleys.iat[0, 0]:
        ex2_actual_lows.append((ex2_df_lows_valleys.iat[0, 0], ex2_df_lows_valleys.iat[0, 1]))

        ex2_actual_highs.append((ex2_df_highs_peaks.iat[0, 0], ex2_df_highs_peaks.iat[0, 1]))
        ex2_extrema.append((ex2_df_highs_peaks.iat[0, 0], ex2_df_highs_peaks.iat[0, 1]))


    ex2_df_highs_peaks_and_lows_valleys = ex2_df_highs_peaks_and_lows_valleys[ex2_df_highs_peaks_and_lows_valleys['Date'] > ex2_extrema[-1][0]]

    for i in range(len(ex2_df_highs_peaks_and_lows_valleys)):
        # Add Extrema High if previous is Low
        if (((ex2_df_highs_peaks_and_lows_valleys.iloc[i][['Date']] == ex2_df_highs_peaks[['Date']]).all(axis=1).any()) and (ex2_extrema[-1][0] == ex2_actual_lows[-1][0]) and (ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Date'] > ex2_extrema[-1][0])):
            ex2_actual_highs.append((ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))
            ex2_extrema.append((ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))

        # Add Extrema Low if previous is High
        if (((ex2_df_highs_peaks_and_lows_valleys.iloc[i][['Date']] == ex2_df_lows_valleys[['Date']]).all(axis=1).any()) and (ex2_extrema[-1][0] == ex2_actual_highs[-1][0]) and (ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Date'] > ex2_extrema[-1][0])):
            ex2_actual_lows.append((ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))
            ex2_extrema.append((ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Date'], ex2_df_highs_peaks_and_lows_valleys.iloc[i]['Extremum']))


    # Convert ex2_df_extrema to DataFrame
    ex2_df_extrema = pd.DataFrame(ex2_extrema, columns=['Date', 'Extremum'])
    ex2_df_extrema = ex2_df_extrema.sort_values(by='Date')
    # print(ex2_df_extrema)



    # STRATEGY 1

    if strategy == 1:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 2

    if strategy == 2:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 3

    if strategy == 3:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 4

    if strategy == 4:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 5

    if strategy == 5:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 6

    if strategy == 6:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 7

    if strategy == 7:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 8

    if strategy == 8:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 9

    if strategy == 9:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 10

    if strategy == 10:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 11

    if strategy == 11:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 12

    if strategy == 12:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 13

    if strategy == 13:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 14

    if strategy == 14:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 15

    if strategy == 15:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 16

    if strategy == 16:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 17

    if strategy == 17:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 18

    if strategy == 18:
        B_trading_points = []
        S_trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost
        win_ratio = won / won_lost
        win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 19

    if strategy == 19:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 20

    if strategy == 20:
        trading_points = []

        for i in range(2, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-3]['Extremum'] > current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-3]['Extremum'] < current_ex1_df_extrema.iloc[-1]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 21

    if strategy == 21:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 22

    if strategy == 22:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 23

    if strategy == 23:
        B_trading_points = []
        S_trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINT6
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 24

    if strategy == 24:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 25

    if strategy == 25:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 26

    if strategy == 26:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 27

    if strategy == 27:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 28

    if strategy == 28:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 29

    if strategy == 29:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 30

    if strategy == 30:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 31

    if strategy == 31:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 32

    if strategy == 32:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 33

    if strategy == 33:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0

        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'

    

    # STRATEGY 34

    if strategy == 34:
        B_trading_points = []
        S_trading_points = []
        
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 1 POINT
                if len(current_ex1_df_extrema) >= 3:
                    if ((current_ex1_df_extrema.iloc[-3]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-4]['Extremum'] > current_ex1_df_extrema.iloc[-2]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-4]['Extremum'] < current_ex1_df_extrema.iloc[-2]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            S_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Sell'))

                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            B_trading_points.append((data.index[i], data.iloc[i]['Open'], 'Close Buy'))


        B_trading_points = pd.DataFrame(B_trading_points, columns=['Date', 'Open', 'Order'])
        S_trading_points = pd.DataFrame(S_trading_points, columns=['Date', 'Open', 'Order'])
        B_trading_points = B_trading_points.sort_values(by='Date')
        S_trading_points = S_trading_points.sort_values(by='Date')

        B_won = 0
        B_lost = 0
        B_order = 0

        S_won = 0
        S_lost = 0
        S_order = 0
        
        # BUY
        for i in range(len(B_trading_points)):
            if B_trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = B_trading_points.iloc[i+1:].index[B_trading_points.iloc[i+1:]['Order'] == 'Close Buy'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    # WON
                    if B_trading_points.iloc[i]['Open'] < B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_won += 1
                        B_order += 1
                        i = next_close_buy_index
                    # LOST
                    if B_trading_points.iloc[i]['Open'] >= B_trading_points.iloc[next_close_buy_index]['Open']:
                        B_lost += 1
                        B_order += 1
                        i = next_close_buy_index


        # SELL
        for i in range(len(S_trading_points)):
            if S_trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = S_trading_points.iloc[i+1:].index[S_trading_points.iloc[i+1:]['Order'] == 'Close Sell'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if S_trading_points.iloc[i]['Open'] > S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_won += 1
                        S_order += 1
                        i = next_close_sell_index
                    # LOST
                    if S_trading_points.iloc[i]['Open'] <= S_trading_points.iloc[next_close_sell_index]['Open']:
                        S_lost += 1
                        S_order += 1
                        i = next_close_sell_index


        won = B_won + S_won
        lost = B_lost + S_lost

        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 35

    if strategy == 35:
        trading_points = []

        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'



    # STRATEGY 36

    if strategy == 36:
        trading_points = []
    
        for i in range(1, len(data.index)):
            if ((data.index[i-2] == ex1_df_extrema[['Date']]).all(axis=1).any()):
                current_ex1_df_extrema = ex1_df_extrema[(ex1_df_extrema['Date'] <= data.index[i-2])]

                # 2 POINTS
                if len(current_ex1_df_extrema) >= 5:
                    if ((current_ex1_df_extrema.iloc[-5]['Date'] == ex2_df_extrema[['Date']]).all(axis=1).any()):
                        # LOW (UPTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['Low'] and current_ex1_df_extrema.iloc[-6]['Extremum'] > current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Sell'))

                        # HIGH (DOWNTREND)
                        if current_ex1_df_extrema.iloc[-1]['Extremum'] == data.iloc[i-2]['High'] and current_ex1_df_extrema.iloc[-6]['Extremum'] < current_ex1_df_extrema.iloc[-4]['Extremum']:
                            trading_points.append((data.index[i], data.iloc[i]['Open'], 'Buy'))


        trading_points = pd.DataFrame(trading_points, columns=['Date', 'Open', 'Order'])
        trading_points = trading_points.sort_values(by='Date')

        won = 0
        lost = 0
        order = 0

        for i in range(len(trading_points)):
            # BUY
            if trading_points.iloc[i]['Order'] == 'Buy':
                next_close_buy_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Sell'].tolist()

                if len(next_close_buy_index) > 0:
                    next_close_buy_index = next_close_buy_index[0]
                    if trading_points.iloc[i]['Open'] < trading_points.iloc[next_close_buy_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_buy_index
                    # LOST
                    if trading_points.iloc[i]['Open'] >= trading_points.iloc[next_close_buy_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_buy_index

            # SELL
            if trading_points.iloc[i]['Order'] == 'Sell':
                next_close_sell_index = trading_points.iloc[i+1:].index[trading_points.iloc[i+1:]['Order'] == 'Buy'].tolist()

                if len(next_close_sell_index) > 0:
                    next_close_sell_index = next_close_sell_index[0]
                    # WON
                    if trading_points.iloc[i]['Open'] > trading_points.iloc[next_close_sell_index]['Open']:
                        won += 1
                        order += 1
                        i = next_close_sell_index
                    # LOST
                    if trading_points.iloc[i]['Open'] <= trading_points.iloc[next_close_sell_index]['Open']:
                        lost += 1
                        order += 1
                        i = next_close_sell_index


        won_lost = won + lost

        if won_lost == 0:
            win_ratio = 0

        if not won_lost ==0:
            win_ratio = won / won_lost
            win_ratio = f'{win_ratio * 100:.2f}%'


    return data, ex1_df_extrema, ex2_df_extrema, won, lost, won_lost, win_ratio



def main():
    mt5.initialize()

    for i in range(21, 37):
        strategy = i
        print('Strategy #', i)
        time_frame = 'H1'
        symbols = ["AAPL.NAS", "MSFT.NAS", "AMZN.NAS", "NVDA.NAS", "XOM.NYSE", "AVGO.NAS", "GOOG.NAS", "TSLA.NAS", "UNH.NYSE", "JPM.NYSE",
        "EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "AUDCAD",
        "BRENT_G4", "WTI_G4", "XBRUSD", "XTIUSD", "XNGUSD", "Cocoa_H4", "Coffee_H4", "Corn_H4", "Cotton_H4", "OJ_F4", "Sbean_F4", "Sugar_H4", "Wheat_H4",
        "AUS200", "DE40", "F40", "JP225", "STOXX50", "UK100", "US30", "US500", "USTEC", "CA60", "CHINA50", "CHINAH", "ES35", "HK50", "IT40", "MidDE50", "NETH25", "NOR25", "SA40", "SE30", "SWI20", "TecDE30", "US2000", "VIX_Z3",
        # "DXY_H4",
        "EURBBL_H4", "EURBND_H4", "EURSCA_H4", "ITB10Y_H4", "JGB10Y_H4", "UKGB_H4", "UST05Y_H4", "UST10Y_H4",
        # "UST30Y_H4",
        "BCHUSD", "BTCUSD", "DOTUSD", "DSHUSD", "EOSUSD", "ETHUSD", "LNKUSD", "LTCUSD", "XLMUSD", "XRPUSD", "ADAUSD", "BNBUSD", "DOGUSD", "UNIUSD", "XTZUSD", "SOLUSD", "MTCUSD", "KSMUSD"]
        final_data = pd.DataFrame(columns=['Symbol', 'Won', 'Lost', 'Sum', 'Win Ratio'])

        for symbol in symbols:
            data, ex1_df_extrema, ex2_df_extrema, won, lost, won_lost, win_ratio = extrema(symbol, time_frame, strategy)
            print('Symbol:', symbol)
            print('Won:', won)
            print('Lost:', lost)
            print('Sum:', won_lost)
            print('Win Ratio:', win_ratio)
            print('=================================================')
            print()
            print()
            print()

            df_result = pd.DataFrame({'Symbol': [symbol], 'Start Date': [data.index[0]], 'Last Date': [data.index[-1]], 'Won': [won], 'Lost': [lost], 'Sum': [won_lost], 'Win Ratio': [win_ratio]})
            final_data = pd.concat([final_data, df_result], ignore_index=True)


        # Export DataFrame to Excel
        final_data.to_excel(f'Strategy {strategy} Timeframe {time_frame}.xlsx', index=False)



if __name__ == '__main__':
    main()