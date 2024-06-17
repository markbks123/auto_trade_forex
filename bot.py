import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Initialize MT5 and login
mt5.initialize()
account =  "" # Replace with your account number
password = ""  # Replace with your password
server = ""  # Replace with your server

if not mt5.login(account, password, server):
    print("Failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
    mt5.shutdown()
    exit()

# Function to fetch historical data
def fetch_data(symbol, start, end, timeframe=mt5.TIMEFRAME_H1):
    print(f"Fetching data for {symbol} from {start} to {end}")
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    
    # Debug output to inspect rates
    print(f"Rates type: {type(rates)}, Length: {len(rates) if rates is not None else 'None'}")
    
    if rates is None or len(rates) == 0:
        raise ValueError("No data retrieved from MT5")

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    print(f"Data columns: {df.columns.tolist()}")  # Debug output to inspect columns

    # Ensure 'time' column exists and convert to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    else:
        raise KeyError("'time' column is missing in the data returned from MT5")

    return df

# Prepare data
start = datetime.datetime(2018, 1, 1)  # Adjust start date
end = datetime.datetime(2024, 6, 14)  # Adjust end date
symbol = "EURUSD"

try:
    df = fetch_data(symbol, start, end)
except ValueError as e:
    print(e)
    mt5.shutdown()
    exit()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

# Create training and test sets
training_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:int(training_data_len), :]
test_data = scaled_data[int(training_data_len):, :]

# Create dataset function
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

#Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predict future prices
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(np.mean(((predictions - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2)))
print(f'RMSE: {rmse}')


def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return rsi

# Function to place a trade with stop loss and take profit
def place_trade(symbol, action, volume, stop_loss=None, take_profit=None):
    mt5.initialize()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick data for {symbol}")
        return None

    current_price = (mt5.symbol_info(symbol).ask + mt5.symbol_info(symbol).bid ) / 2
    if current_price == 0.0:
        print(f"Invalid current price for {symbol}")
        return None
    if action == "buy":
        price = tick.bid
        order_type =  mt5.ORDER_TYPE_BUY
    else:
        order_type =  mt5.ORDER_TYPE_SELL
        price = tick.ask


    print(mt5.symbol_info(symbol).point)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": f"{action} order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK  # Change to FOK (Fill or Kill)
    }

    if stop_loss:
        if action == "buy":
            request["sl"] = price - stop_loss * mt5.symbol_info(symbol).point
        else:
            request["sl"] = price + stop_loss * mt5.symbol_info(symbol).point

    if take_profit:
        if action == "buy":
            request["tp"] = price + take_profit * mt5.symbol_info(symbol).point
        else:
            request["tp"] = price - take_profit * mt5.symbol_info(symbol).point

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to place {action} order: {result.comment}")
    else:
        print(f"{action.capitalize()} order placed successfully at {price}, SL: {request.get('sl', 'None')}, TP: {request.get('tp', 'None')}")

    return result
# Run trading bot
def run_trading_bot(symbol, model, scaler, stop_loss_pct=0.01, take_profit_pct=0.02, time_step=60):
    while True:
        print("Fetching latest data and making predictions...")

        # Fetch the latest data
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=60)
        try:
            df = fetch_data(symbol, start, end)
        except ValueError as e:
            print(f"Error fetching data: {e}")
            continue

        scaled_data = scaler.transform(df['close'].values.reshape(-1, 1))

        # Prepare the data for prediction
        X_pred = scaled_data[-time_step:].reshape(1, time_step, 1)
        prediction = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        # Fetch current price
        tick = mt5.symbol_info_tick(symbol)
        current_rsi= calculate_rsi(df)



        if tick is None:
            print(f"Failed to get tick data for {symbol}")
            continue

        current_price = (  tick.ask +   tick.bid ) / 2
        print(f"Current price: {current_price}, Predicted price: {predicted_price}")

        # Check if current_price is valid
        if current_price == 0.0:
            print("Invalid current price, skipping this iteration.")
            continue

        # Calculate stop loss and take profit based on predicted price
        if predicted_price > current_price:
            if current_rsi < 30:  # Oversold condition
                sl = 0.001 * 2  # Example: 20 pips
                tp = 0.001 * 8  # Example: 80 pips
            else:
                sl = 0.001 * 1  # Example: 10 pips
                tp = 0.001 * 2  # Example: 20 pips
            place_trade(symbol, "buy", 0.1, stop_loss=sl, take_profit=tp)
        else:
            if current_rsi > 70:  # Overbought condition
                sl = 0.001 * 2  # Example: 20 pips
                tp = 0.001 * 8  # Example: 80 pips
            else:
                sl = 0.001 * 1  # Example: 10 pips
                tp = 0.001 * 2  # Example: 20 pips
            place_trade(symbol, "sell", 0.1, stop_loss=sl, take_profit=tp)
        # Wait for the next interval (5 minutes)
        time.sleep(3600)


# run_trading_bot(symbol, model, scaler, stop_loss_pct=0.01, take_profit_pct=0.02)
rs = calculate_rsi(df)


print(df['RSI'].iloc[-1])


mt5.shutdown()
