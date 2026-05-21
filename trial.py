import argparse
import numpy as np
import pandas as pd
import nni
import math
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        for _ in range(num_layers):
            attention_output = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=units)(x, x)
            attention_output = layers.Dropout(dropout)(attention_output)
            out1 = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
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
                  metrics=['mae', 'mape'])
    return model

def sanitize_metric(val):
    """ Prevents NaN or Inf from crashing the NNI JSON reporter """
    if val is None or math.isnan(val) or math.isinf(val):
        return 99999.0 
    return float(val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to training dataset')
    parser.add_argument('--colname', type=str, default=None, help='column name if csv has header')
    args = parser.parse_args()

    params = nni.get_next_parameter()
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
    
    window = int(params['window_size'])

    # =====================================================================
    # 1. LOAD TRAINING DATA
    # =====================================================================
    df_train = pd.read_csv(args.data)
    if args.colname is not None:
        series_train = df_train[args.colname].values
    else:
        series_train = df_train.iloc[:, 0].values

    series_train = series_train.astype('float32')
    X_train_full, y_train_full = create_sequences(series_train, window)

    # CRITICAL: Drop the last 200 sequences from the training data.
    # If the training file is "local_plus_tt.csv" or "local_only.csv", this ensures
    # the model never sees the evaluation data during training (preventing data leakage).
    # If the file is "ali_only.csv", we just drop 200 Alibaba samples, which is fine.
    X_train = X_train_full[:-200]
    y_train = y_train_full[:-200]

    # =====================================================================
    # 2. LOAD UNIVERSAL VALIDATION DATA (ALWAYS LOCAL_ONLY)
    # =====================================================================
    # We hardcode loading local_only.csv so EVERY experiment is evaluated on it.
    df_val = pd.read_csv("./local_only.csv")
    if args.colname is not None:
        series_val = df_val[args.colname].values
    else:
        series_val = df_val.iloc[:, 0].values

    series_val = series_val.astype('float32')
    X_val_full, y_val_full = create_sequences(series_val, window)

    # Take EXACTLY the last 200 sequences of the local dataset for validation
    X_val = X_val_full[-200:]
    y_val = y_val_full[-200:]

    # =====================================================================

    model = build_model(params, input_shape=X_train.shape[1:])

    epochs = int(params['epochs'])
    batch_size = int(params['batch_size'])

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    for epoch in range(1, epochs + 1):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=1, batch_size=batch_size, callbacks=[early_stop], verbose=0)
        val_loss = sanitize_metric(history.history['val_loss'][-1])
        nni.report_intermediate_result(val_loss)

    try:
        results = model.evaluate(X_val, y_val, verbose=0)
        final_val_loss = sanitize_metric(results[0])  # mse
        final_mae = sanitize_metric(results[1])
        final_mape = sanitize_metric(results[2])
        
        # Calculate R2 score
        y_pred = model.predict(X_val, verbose=0)
        final_r2 = sanitize_metric(r2_score(y_val.flatten(), y_pred.flatten()))
        
    except Exception as e:
        print("Error during final evaluation:", e)
        final_val_loss = 99999.0
        final_mae = 99999.0
        final_mape = 99999.0
        final_r2 = -99999.0

    nni.report_final_result({
        'default': final_val_loss, # Changed from final_mape to final_val_loss (MSE)
        'mse': final_val_loss,
        'mae': final_mae,
        'mape': final_mape,
        'r2': final_r2
    })
    
    print('============ Final Results ============')
    print(f'Final MSE: {final_val_loss}')
    print(f'Final MAE: {final_mae}')
    print(f'Final MAPE: {final_mape}')
    print(f'Final R2: {final_r2}')
    print('=======================================')

if __name__ == '__main__':
    main()