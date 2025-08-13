import argparse
import numpy as np
import pandas as pd
import nni
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# create_sequences function to prepare time series data for training
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y

def build_model(params, input_shape):
    model_type = params.get('model_type', 'LSTM')
    units = int(params.get('units', 32))
    num_layers = int(params.get('num_layers', 1))
    dropout = float(params.get('dropout', 0.0))
    lr = float(params.get('lr', 1e-3))
    activation = params.get('activation', 'relu')
    kernel_size = int(params.get('kernel_size', 3))
    attention_heads = int(params.get('attention_heads', 4))

    inputs = keras.Input(shape=input_shape)
    x = inputs

    if model_type in ['LSTM', 'GRU', 'RNN']:
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            if model_type == 'LSTM':
                x = layers.LSTM(units, return_sequences=return_sequences)(x)
            elif model_type == 'GRU':
                x = layers.GRU(units, return_sequences=return_sequences)(x)
            else:  # RNN simple
                x = layers.SimpleRNN(units, return_sequences=return_sequences)(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

    elif model_type == 'CNN':
        for _ in range(num_layers):
            x = layers.Conv1D(filters=units, kernel_size=kernel_size, padding='same', activation=activation)(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
        x = layers.GlobalAveragePooling1D()(x)

    elif model_type == 'TCN':
        # Temporal Convolutional Network approximation with dilated Conv1D
        dilation_rates = [2**i for i in range(num_layers)]
        for dilation_rate in dilation_rates:
            x = layers.Conv1D(filters=units,
                              kernel_size=kernel_size,
                              padding='causal',
                              dilation_rate=dilation_rate,
                              activation=activation)(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
        x = layers.GlobalAveragePooling1D()(x)

    elif model_type == 'Transformer':
        # Simplified transformer encoder block for time series
        # Positional encoding could be added, but skipped here for brevity
        for _ in range(num_layers):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=units)(x, x)
            attention_output = layers.Dropout(dropout)(attention_output)
            out1 = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            # Feed-forward network
            ffn = layers.Dense(units*4, activation=activation)(out1)
            ffn = layers.Dense(units)(ffn)
            ffn = layers.Dropout(dropout)(ffn)
            x = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
        x = layers.GlobalAveragePooling1D()(x)

    elif model_type == 'MLP':
        x = layers.Flatten()(x)
        for _ in range(num_layers):
            x = layers.Dense(units, activation=activation)(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae', 'mape', 'rmse'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to csv with single column of series')
    # for now, the expiriments take a single column from the csv file as it is the only colums
    parser.add_argument('--colname', type=str, default=None, help='column name if csv has header')
    args = parser.parse_args()

    params = nni.get_next_parameter()
    # Fill missing params with defaults to avoid errors
    params = {
        'model_type': params.get('model_type', 'LSTM'),
        'units': params.get('units', 32),
        'num_layers': params.get('num_layers', 1),
        'dropout': params.get('dropout', 0.0),
        'lr': params.get('lr', 1e-3),
        'window_size': params.get('window_size', 50),
        'batch_size': params.get('batch_size', 64),
        'epochs': int(params.get('epochs', 10)),
        'activation': params.get('activation', 'relu'),
        'kernel_size': params.get('kernel_size', 3),
        'attention_heads': params.get('attention_heads', 4)
    }

    df = pd.read_csv(args.data)
    if args.colname is not None:
        series = df[args.colname].values
    else:
        series = df.iloc[:, 0].values

    series = series.astype('float32')

    window = int(params['window_size'])
    X, y = create_sequences(series, window)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(params, input_shape=X_train.shape[1:])

    epochs = int(params['epochs'])
    batch_size = int(params['batch_size'])

    for epoch in range(1, epochs + 1):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=1, batch_size=batch_size, verbose=0)
        val_loss = history.history['val_loss'][-1]
        nni.report_intermediate_result(val_loss)

    try:
        results = model.evaluate(X_val, y_val, verbose=0)
        final_val_loss = float(results[0])  # mse
        final_mae = float(results[1])
        final_mape = float(results[2])
    except Exception as e:
        print("Error during final evaluation:", e)
        final_val_loss = float('inf')
        final_mae = float('inf')
        final_mape = float('inf')

    nni.report_final_result({
        'default': final_mape,
        'mse': final_val_loss,
        'mae': final_mae,
        'mape': final_mape
    })

if __name__ == '__main__':
    main()
