import keras.callbacks
import numpy
import talib
from keras.layers import Dense, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D
from keras.layers import LSTM
from keras.models import Sequential
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("myApp").getOrCreate()

sdf = spark.read.parquet('/Users/dshurick/data/cbpro/data/BTC-USD/')

resampled = sdf.withColumn('time_slice', F.window('time', '6 hours')).select(
    '*',
    F.first('open').over(
        Window.partitionBy('time_slice').orderBy('time')).alias('first_open'),
    F.first('close').over(
        Window.partitionBy('time_slice').orderBy(
            F.desc('time'))).alias('last_close'),
).groupBy('time_slice').agg(
    F.first('first_open').alias('open'),
    F.max('high').alias('high'),
    F.min('low').alias('low'),
    F.first('last_close').alias('close'),
    F.sum('volume').alias('volume'),
).orderBy('time_slice')

resampled_pdf = resampled.toPandas()

resampled_pdf['TYPPRICE'] = talib.TYPPRICE(
    high=resampled_pdf.high,
    low=resampled_pdf.low,
    close=resampled_pdf.close,
)

resampled_pdf['LOGPRICE'] = numpy.log(resampled_pdf.TYPPRICE)

resampled_array = resampled_pdf.values

cutoff = int(len(resampled_array) * 0.2)

train = resampled_array[:-cutoff, [-2]]
test = resampled_array[-cutoff:, [-2]]

# scaler = MinMaxScaler((0, 1))

# train = scaler.fit_transform(train)
# test = scaler.transform(test)

import sklearn.metrics


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['{:+0.1%}'.format(s) for s in scores])
    print('{}: [{:+0.1%}] {}'.format(name, score, s_scores))


def evaluate_model(model, test_x, test_y):
    n_input = test_x.shape[1]
    n_out = test_y.shape[1]

    predictions = list()
    for ii in range(len(test_x)):

        input_x = test_x[[ii]]
        yhat = model.predict(input_x, verbose=0).flatten()
        ytrue = test_y[ii].flatten()

        # Dumb model is just to forecast the most recent observation going forward
        baseline_prediction = numpy.repeat(input_x.flatten()[-1], len(ytrue))
        predictions.append((yhat, ytrue, baseline_prediction))

    predictions = numpy.array(predictions)

    forecast_bias = predictions[:, 0, :] - predictions[:, 1, :]
    #     total_rmse = numpy.sqrt(numpy.power(forecast_bias, 2).mean())
    #     rmse_by_day = numpy.power(numpy.power(forecast_bias, 2).mean(axis=0), 0.5)
    total_rmse = (numpy.abs(forecast_bias)).mean()
    rmse_by_day = (numpy.abs(forecast_bias)).mean(axis=0)

    baseline_bias = predictions[:, 2, :] - predictions[:, 1, :]
    #     baseline_total_rmse = numpy.sqrt(numpy.power(baseline_bias, 2).mean())
    #     baseline_rmse_by_day = numpy.power(
    #         numpy.power(baseline_bias, 2).mean(axis=0), 0.5)
    baseline_total_rmse = (numpy.abs(baseline_bias)).mean()
    baseline_rmse_by_day = (numpy.abs(baseline_bias)).mean(axis=0)

    rmse_pct_chng = (total_rmse - baseline_total_rmse) / baseline_total_rmse
    rmse_by_day_pct_chng = (
        rmse_by_day - baseline_rmse_by_day) / baseline_rmse_by_day

    return baseline_total_rmse, baseline_rmse_by_day, total_rmse, rmse_by_day


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(train)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(train):
            X.append(train[in_start:in_end, :])
            y.append(train[in_end:out_end, :])
        # move along one time step
        in_start += 1
    return numpy.array(X), numpy.array(y)


# # LSTM Model With Univariate Input and Vector Output

train_x, train_y = to_supervised(train, n_input=14, n_out=7)
train_y = train_y.reshape((train_y.shape[0:2]))

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
)

epochs = 200
batch_size = 2**5

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))
model.compile(loss='mae', optimizer='adam')

# fit network
model.fit(
    train_x,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping],
)

test_x, test_y = to_supervised(test, n_input=14, n_out=7)
test_y = test_y.reshape((test_y.shape[0:2]))

baseline_total_rmse, baseline_rmse_by_day, total_rmse, rmse_by_day = evaluate_model(
    model, test_x, test_y)


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['{:.4f}'.format(s) for s in scores])
    print('{}: [{:.4f}] {}'.format(name, score, s_scores))


summarize_scores('BASELINE ', baseline_total_rmse, baseline_rmse_by_day)
summarize_scores('MODEL    ', total_rmse, rmse_by_day)

# # Encoder-Decoder LSTM Model With Univariate Input

n_timesteps = train_x.shape[1]
n_features = train_x.shape[2]
n_outputs = train_y.shape[1]

# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    restore_best_weights=True,
)

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mape', optimizer='adam')

# fit network
model.fit(
    train_x,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping],
)

test_x, test_y = to_supervised(test, n_input=7, n_out=14)
test_y = test_y.reshape((test_y.shape[0:2]))

score, scores = evaluate_model(model, test_x, test_y)

summarize_scores('lstm', score, scores)

# # CNN-LSTM Encoder-Decoder Model With Univariate Input

# In[ ]:

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]

# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

model = Sequential()
model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')

# fit network
model.fit(
    train_x,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping],
)

test_x, test_y = to_supervised(test, n_input=7, n_out=14)
test_y = test_y.reshape((test_y.shape[0:2]))

score, scores = evaluate_model(model, test_x, test_y)
summarize_scores('lstm', score, scores)

# # ConvLSTM Encoder-Decoder Model With Univariate Input

# In[ ]:


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# In[ ]:

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]

# reshape into subsequences [samples, time steps, rows, cols, channels]
train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))

# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

model = Sequential()
model.add(
    ConvLSTM2D(
        filters=64,
        kernel_size=(1, 4),
        activation='relu',
        input_shape=(n_steps, 1, n_length, n_features)))
model.add(Flatten())
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))

train_x, train_y = to_supervised(train, n_input=14, n_out=7)

verbose = 1
epochs = 100
batch_size = 2**4

n_timesteps = train_x.shape[1]
n_features = train_x.shape[2]
n_outputs = train_y.shape[1]

# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    restore_best_weights=True,
)

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')

# fit network
model.fit(
    train_x,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=verbose,
    validation_split=0.2,
    callbacks=[early_stopping],
)

score, scores = evaluate_model(model, test, n_input=14, n_out=7)

summarize_scores('lstm', score, scores)

scaler.inverse_transform(ytrue.reshape(-1, 1))

scaler.inverse_transform(yhat.reshape(-1, 1))

n_steps, n_length = 2, 7
# define the total days to use as input
n_input = n_length * n_steps

train_x, train_y = to_supervised(train, n_input)

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]

# reshape into subsequences [samples, time steps, rows, cols, channels]
train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

model = Sequential()
model.add(
    ConvLSTM2D(
        filters=64,
        kernel_size=(1, 3),
        activation='relu',
        input_shape=(n_steps, 1, n_length, n_features)))
model.add(Flatten())
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')

model.fit(
    train_x,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=verbose,
    validation_split=0.2,
    callbacks=[early_stopping],
)

test_x, test_y = to_supervised(test, n_input=14, n_out=7)
test_x = test_x.reshape((test_x.shape[0], n_steps, 1, n_length, n_features))
test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

score, scores = evaluate_model(model, test_x, test_y, n_input=14, n_out=7)

summarize_scores('lstm', score, scores)

test_x[1].flatten()

test_y[1].flatten()

pdf = sdf.toPandas()
pdf.sort_values(['product_id', 'time'], inplace=True)


def apply_talib_fns(pdf, dropna=True):

    # Price Transform
    pdf['TYPPRICE'] = talib.TYPPRICE(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['AVGPRICE'] = talib.AVGPRICE(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['MEDPRICE'] = talib.MEDPRICE(
        high=pdf.high,
        low=pdf.low,
    )

    # Cycle Indicators

    pdf['HT_DCPERIOD'] = talib.HT_DCPERIOD(pdf.TYPPRICE)
    pdf['HT_DCPHASE'] = talib.HT_DCPHASE(pdf.TYPPRICE)
    pdf['HT_PHASOR_inphase'], pdf['HT_PHASOR_quadrature'] = talib.HT_PHASOR(
        pdf.TYPPRICE)
    pdf['HT_SINE_sine'], pdf['HT_SINE_leadsine'] = talib.HT_SINE(pdf.TYPPRICE)
    pdf['HT_TRENDMODE'] = talib.HT_TRENDMODE(pdf.TYPPRICE)

    # Momentum Indicators

    pdf['ADX'] = talib.ADX(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['ADXR'] = talib.ADXR(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['APO'] = talib.APO(pdf.TYPPRICE)

    pdf['AROON_down'], pdf['AROON_up'] = talib.AROON(
        high=pdf.high,
        low=pdf.low,
    )

    pdf['AROONOSC'] = talib.AROONOSC(
        high=pdf.high,
        low=pdf.low,
    )

    pdf['BOP'] = talib.BOP(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['CCI'] = talib.CCI(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['CMO'] = talib.CMO(pdf.TYPPRICE)

    pdf['DX'] = talib.DX(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['MACD_macd'], pdf['MACD_signal'], pdf['MACD_hist'] = talib.MACD(
        pdf.TYPPRICE)

    pdf['MACDEXT_macd'], pdf['MACDEXT_signal'], pdf[
        'MACDEXT_hist'] = talib.MACD(pdf.TYPPRICE)

    pdf['MFI'] = talib.MFI(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
        volume=pdf.volume,
    )

    pdf['MINUS_DI'] = talib.MINUS_DI(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['MINUS_DM'] = talib.MINUS_DM(
        high=pdf.high,
        low=pdf.low,
    )

    pdf['MOM'] = talib.MOM(pdf.TYPPRICE)

    pdf['PLUS_DI'] = talib.PLUS_DI(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['PLUS_DM'] = talib.PLUS_DM(
        high=pdf.high,
        low=pdf.low,
    )

    pdf['PPO'] = talib.PPO(pdf.TYPPRICE)
    pdf['ROC'] = talib.ROC(pdf.TYPPRICE)
    pdf['ROCP'] = talib.ROCP(pdf.TYPPRICE)
    pdf['ROCR'] = talib.ROCR(pdf.TYPPRICE)
    pdf['RSI'] = talib.RSI(pdf.TYPPRICE)

    pdf['STOCH_slowk'], pdf['STOCH_slowd'] = talib.STOCH(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['STOCHF_slowk'], pdf['STOCHF_slowd'] = talib.STOCHF(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['STOCHRSI_slowk'], pdf['STOCHRSI_slowd'] = talib.STOCHRSI(pdf.TYPPRICE)

    pdf['TRIX'] = talib.TRIX(pdf.TYPPRICE)

    pdf['ULTOSC'] = talib.ULTOSC(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['WILLR'] = talib.WILLR(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    # Volatility Indicators

    pdf['ATR'] = talib.ATR(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['TRANGE'] = talib.TRANGE(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    # Volume Indicators

    pdf['AD'] = talib.AD(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
        volume=pdf.volume,
    )

    pdf['ADOSC'] = talib.ADOSC(
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
        volume=pdf.volume,
    )

    pdf['OBV'] = talib.OBV(
        pdf.TYPPRICE,
        pdf.volume,
    )

    # Pattern Recognition

    pdf['CDL2CROWS'] = talib.CDL2CROWS(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['CDL3INSIDE'] = talib.CDL3INSIDE(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    pdf['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(
        open=pdf.open,
        high=pdf.high,
        low=pdf.low,
        close=pdf.close,
    )

    # Overlap Studies
    pdf['upperband'], pdf['middleband'], pdf['lowerband'] = talib.BBANDS(
        pdf.TYPPRICE)

    pdf['DEMA'] = talib.DEMA(pdf.TYPPRICE)
    pdf['EMA'] = talib.EMA(pdf.TYPPRICE)
    pdf['HT_TRENDLINE'] = talib.HT_TRENDLINE(pdf.TYPPRICE)
    pdf['KAMA'] = talib.KAMA(pdf.TYPPRICE)
    pdf['MA'] = talib.MA(pdf.TYPPRICE)
    pdf['MAMA_mama'], pdf['MAMA_fama'] = talib.MAMA(pdf.TYPPRICE)
    # pdf['MAVP'] = talib.MAVP(pdf.TYPPRICE)
    pdf['MIDPOINT'] = talib.MIDPOINT(pdf.TYPPRICE)

    pdf['MIDPRICE'] = talib.MIDPRICE(
        high=pdf.high,
        low=pdf.low,
    )
    pdf['SAR'] = talib.SAR(
        high=pdf.high,
        low=pdf.low,
    )
    pdf['SAREXT'] = talib.SAREXT(
        high=pdf.high,
        low=pdf.low,
    )
    pdf['SMA'] = talib.SMA(pdf.TYPPRICE)
    pdf['T3'] = talib.T3(pdf.TYPPRICE)
    pdf['TEMA'] = talib.TEMA(pdf.TYPPRICE)
    pdf['TRIMA'] = talib.TRIMA(pdf.TYPPRICE)
    pdf['WMA'] = talib.WMA(pdf.TYPPRICE)

    if dropna:
        pdf.dropna(inplace=True)


apply_talib_fns(resampled_pdf)

resampled_pdf

pdf.dropna(inplace=True)

pdf.to_pickle(
    '/Users/dshurick/data/bitcoin_price_history.pkl',
    compression='gzip',
)

# In[ ]:


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=5)

for train_ind, test_ind in tscv.split(pdf):
    pass

train, test = pdf.iloc[train_ind].values, pdf.iloc[test_ind].values

n_input, n_out = 14, 7

X, y = list(), list()
in_start = 0
# step over the entire history one time step at a time
for _ in range(len(train)):
    # define the end of the input sequence
    in_end = in_start + n_input
    out_end = in_end + n_out
    # ensure we have enough data for this instance
    if out_end < len(train):
        x_input = train[in_start:in_end, 7]
        x_input = x_input.reshape((len(x_input), 1))
        X.append(x_input)
        y.append(train[in_end:out_end, 7])
    # move along one time step
    in_start += 1

import numpy

train_x, train_y = numpy.array(X), numpy.array(y)

from keras.layers import (
    Dense,
    RepeatVector,
    TimeDistributed,
)
from keras.layers import LSTM
from keras.models import Sequential

2**7

verbose, epochs, batch_size = 1, 20, 2**7
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]
# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')

model.fit(
    train_x,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=verbose,
)

# In[ ]:

n_input = 7
train_x, train_y = to_supervised(train, n_input)

indices = tscv.split(pdf)

train_ind, test_ind = next(indices)


def split_sequence(X, y=None, n_steps_in=1, n_steps_out=1):

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)


#     if y is not None and len(y.shape) == 1:
#         y = y.reshape(-1, 1)

    X_out, y_out = list(), list()

    for i in range(len(X)):

        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence
        if out_end_ix > len(X):
            break

        if y is None:
            # gather input and output parts of the pattern
            if len(X.shape) > 1:
                seq_x, seq_y = X[i:end_ix, :], X[end_ix:out_end_ix, :]
            else:
                seq_x, seq_y = X[i:end_ix], X[end_ix:out_end_ix]

        else:
            if len(X.shape) > 1:
                seq_x = X[i:end_ix, :]
            else:
                seq_x = X[i:end_ix]
            if len(y.shape) > 1:
                seq_y = y[end_ix:out_end_ix, :]
            else:
                seq_y = y[end_ix:out_end_ix]

        X_out.append(seq_x)
        y_out.append(seq_y)

    return array(X_out), array(y_out)

pdf_train = pdf.iloc[train_ind]
pdf_test = pdf.iloc[test_ind]

X_train = pdf_train['TYPPRICE'].values
y_train = pdf_train['TYPPRICE'].values

X_test = pdf_test['TYPPRICE'].values
y_test = pdf_test['TYPPRICE'].values

n_steps_in = 12
n_steps_out = 12

X_train_seq, y_train_seq = split_sequence(
    X=X_train,
    y=y_train,
    n_steps_in=n_steps_in,
    n_steps_out=n_steps_out,
)

X_test_seq, y_test_seq = split_sequence(
    X=X_test,
    y=y_test,
    n_steps_in=n_steps_in,
    n_steps_out=n_steps_out,
)

from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
)
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(
    LSTM(100, activation='relu', return_sequences=True, input_shape=(12, 1)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(12))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
    monitor='loss', patience=5, restore_best_weights=True)

# fit model
model.fit(
    X_train_seq,
    y_train_seq,
    epochs=300,
    verbose=1,
    callbacks=[early_stop],
)

y_pred = model.predict(X_test_seq)

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM

register_matplotlib_converters()

from keras.layers import Dense, TimeDistributed, RepeatVector

from keras.callbacks import EarlyStopping

from numpy import array

df = pd.read_csv("/Users/dshurick/Downloads/vix_2011_2019.csv")


def split_sequence(X, y=None, n_steps_in=1, n_steps_out=1):

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if y and len(y.shape) == 1:
        y = y.reshape(-1, 1)

    X_out, y_out = list(), list()

    for i in range(len(X)):

        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence
        if out_end_ix > len(X):
            break

        if y is None:
            # gather input and output parts of the pattern
            if len(X.shape) > 1:
                seq_x, seq_y = X[i:end_ix, :], X[end_ix:out_end_ix, :]
            else:
                seq_x, seq_y = X[i:end_ix], X[end_ix:out_end_ix]

        else:
            if len(X.shape) > 1:
                seq_x = X[i:end_ix, :]
            else:
                seq_x = X[i:end_ix]
            if len(y.shape > 1):
                seq_y = y[end_ix:out_end_ix, :]
            else:
                seq_y = y[end_ix:out_end_ix]

        X_out.append(seq_x)
        y_out.append(seq_y)

    return array(X_out), array(y_out)


n_steps_in = 14

n_steps_out = 1

X, y = split_sequence(
    df.loc[:, 'Open':'Close'].values,
    n_steps_in=n_steps_in,
    n_steps_out=n_steps_out)

n_features = X.shape[2]

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='loss', patience=10)
history = model.fit(
    X,
    y,
    epochs=300,
    batch_size=1,
    verbose=1,
    callbacks=[early_stop],
    shuffle=False)

# In[ ]:

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)

df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Date'], drop=True)
df.head(10)

split_date = pd.Timestamp('2018-01-01')
df = df['Adj Close']
train = df.loc[:split_date]
test = df.loc[split_date:]
plt.figure(figsize=(10, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(pd.DataFrame(train))
test_sc = scaler.transform(pd.DataFrame(test))

X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]

nn_model = Sequential()
nn_model.add(Dense(12, input_dim=1, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = nn_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=1,
    verbose=1,
    callbacks=[early_stop],
    shuffle=False)

y_pred_test_nn = nn_model.predict(X_test)
y_train_pred_nn = nn_model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(
    r2_score(y_train, y_train_pred_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(
    r2_score(y_test, y_pred_test_nn)))

train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)

for s in range(1, 2):
    train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
    test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)

X_train = train_sc_df.dropna().drop('Y', axis=1)
y_train = train_sc_df.dropna().drop('X_1', axis=1)

X_test = test_sc_df.dropna().drop('Y', axis=1)
y_test = test_sc_df.dropna().drop('X_1', axis=1)

X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values

X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print('Train shape: ', X_train_lmse.shape)
print('Test shape: ', X_test_lmse.shape)

lstm_model = Sequential()
lstm_model.add(
    LSTM(
        7,
        input_shape=(1, X_train_lmse.shape[1]),
        activation='relu',
        kernel_initializer='lecun_uniform',
        return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(
    X_train_lmse,
    y_train,
    epochs=100,
    batch_size=1,
    verbose=1,
    shuffle=False,
    callbacks=[early_stop])

y_pred_test_lstm = lstm_model.predict(X_test_lmse)
y_train_pred_lstm = lstm_model.predict(X_train_lmse)
print("The R2 score on the Train set is:\t{:0.3f}".format(
    r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(
    r2_score(y_test, y_pred_test_lstm)))

nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)
lstm_test_mse = lstm_model.evaluate(X_test_lmse, y_test, batch_size=1)
print('NN: %f' % nn_test_mse)
print('LSTM: %f' % lstm_test_mse)

X_test_lmse[0:1]

lstm_y_pred_test = lstm_model.predict(X_test_lmse[0:1])

y_test[0]

lstm_y_pred_test

nn_y_pred_test = nn_model.predict(X_test)
lstm_y_pred_test = lstm_model.predict(X_test_lmse)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_nn, label='NN')
plt.title("NN's Prediction")
plt.xlabel('Observation')
plt.ylabel('Adj Close Scaled')
plt.legend()
plt.show()


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# In[ ]:


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores
