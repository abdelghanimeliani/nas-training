import pandas as pd
import numpy as np
import time
import psutil, os, threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Conv1D, Flatten, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 0. CPU TRACKER (peak + average) - FIXED VERSION
# ============================================================

class CPUTracker:
    def __init__(self, interval=0.1):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.running = False
        self.cpu_percent_list = []
        
        # Initialize CPU percentage measurement
        self.process.cpu_percent(interval=None)

    def _track(self):
        while self.running:
            # Get CPU percentage since last call
            cpu = self.process.cpu_percent(interval=None)
            # Divide by number of CPU cores to get normalized percentage (0-100%)
            cpu_normalized = cpu / psutil.cpu_count()
            self.cpu_percent_list.append(cpu_normalized)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cpu_percent_list:
            max_cpu = max(self.cpu_percent_list)
            avg_cpu = sum(self.cpu_percent_list) / len(self.cpu_percent_list)
        else:
            max_cpu = 0
            avg_cpu = 0
        return max_cpu, avg_cpu

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    df1 = pd.read_csv("dataset1_140.csv")  # TimeTrack 45s
    df2 = pd.read_csv("dataset2_140.csv")  # 5 min dataset A
    df3 = pd.read_csv("dataset3_140.csv")  # 5 min dataset B

    data = {
        "TimeTrack": df1.iloc[:, 0].values,  # Load all available data
        "Dataset2": df2.iloc[:, 0].values,    # Load all available data
        "Dataset3": df3.iloc[:, 0].values     # Load all available data
    }
    return data

# ============================================================
# 2. CREATE SEQUENCES
# ============================================================

def create_sequences(series, window=10):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

# ============================================================
# 3. MODEL BUILDERS
# ============================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(32, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_rnn(input_shape):
    model = Sequential([
        SimpleRNN(32, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_tcn(input_shape):
    # Fixed TCN implementation with padding to maintain sequence length
    model = Sequential([
        Conv1D(32, 3, activation='relu', padding='same', dilation_rate=1, input_shape=input_shape),
        Conv1D(32, 3, activation='relu', padding='same', dilation_rate=2),
        Conv1D(32, 3, activation='relu', padding='same', dilation_rate=4),
        GlobalAveragePooling1D(),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)
    attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = x + attn
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def build_mlp(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model_builders = {
    "LSTM": build_lstm,
    "GRU": build_gru,
    "RNN": build_rnn,
    "CNN": build_cnn,
    "Transformer": build_transformer,
}

# ============================================================
# 4. EXPERIMENT WITH DIFFERENT DATA SIZES - RUN 10 TIMES
# ============================================================

# Define different data sizes to test (keeping 6300/945 ratio)
data_sizes = [
    (300, 45),
    (600, 90),
    (1200, 180),   
    (2400, 360),    
    (4800, 720),
    (9600, 1440),
    (19200, 2880),
    (38400, 5760),
]

NUM_RUNS = 20
WINDOW = 10
EPOCHS = 25
BATCH = 16
EPSILON = 1e-8  # for safe MAPE calculation

# Load all available data first
full_data_dict = load_data()

# Initialize or load existing results
try:
    # Try to load existing results
    df_all_runs = pd.read_csv("avg.csv")
    print("Loaded existing results from avg.csv")
except FileNotFoundError:
    # Create empty DataFrame if file doesn't exist
    df_all_runs = pd.DataFrame()
    print("Starting new experiment - no existing results found")

try:
    for run in range(NUM_RUNS):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{NUM_RUNS}")
        print(f"{'='*60}")
        
        run_results = []
        
        for size_idx, (size1, size2) in enumerate(data_sizes):
            print(f"\nExperiment {size_idx + 1}/{len(data_sizes)} - Data sizes: TimeTrack={size1}, Dataset2/3={size2}")
            
            # Check if we have enough data
            if size1 > len(full_data_dict["TimeTrack"]):
                print(f"Warning: Requested {size1} samples for TimeTrack but only {len(full_data_dict['TimeTrack'])} available")
                size1 = len(full_data_dict["TimeTrack"])
            
            if size2 > len(full_data_dict["Dataset2"]):
                print(f"Warning: Requested {size2} samples for Dataset2 but only {len(full_data_dict['Dataset2'])} available")
                size2 = len(full_data_dict["Dataset2"])
            
            if size2 > len(full_data_dict["Dataset3"]):
                print(f"Warning: Requested {size2} samples for Dataset3 but only {len(full_data_dict['Dataset3'])} available")
                size2 = len(full_data_dict["Dataset3"])
            
            # Create data dictionary with current sizes
            data_dict = {
                "TimeTrack": full_data_dict["TimeTrack"][:size1],
                "Dataset2": full_data_dict["Dataset2"][:size2],
                "Dataset3": full_data_dict["Dataset3"][:size2]
            }
            
            for dataset_name, series in data_dict.items():
                X, y = create_sequences(series, window=WINDOW)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # 80/20 split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                for model_name, builder in model_builders.items():
                    
                    print(f"Training {model_name} on {dataset_name} (size: {len(series)})")

                    try:
                        model = builder(input_shape=(WINDOW, 1))

                        # ---- MEMORY BEFORE ----
                        process = psutil.Process(os.getpid())
                        mem_before = process.memory_info().rss / 1024**2

                        # ---- CPU TRACKING ----
                        cpu_tracker = CPUTracker()
                        cpu_tracker.start()

                        # ---- TIME MEASUREMENT ----
                        start_time = time.time()

                        model.fit(
                            X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH,
                            verbose=0,
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                        )

                        training_time = time.time() - start_time

                        # ---- STOP CPU TRACKING ----
                        max_cpu, avg_cpu = cpu_tracker.stop()

                        # ---- MEMORY AFTER ----
                        mem_after = process.memory_info().rss / 1024**2
                        mem_used = abs(mem_after - mem_before)

                        # ---- EVALUATION ----
                        preds = model.predict(X_test, verbose=0).flatten()

                        mae = mean_absolute_error(y_test, preds)
                        mse = mean_squared_error(y_test, preds)
                        mape = np.mean(np.abs((y_test - preds) / (y_test + EPSILON))) * 100

                        run_results.append({
                            "Run": run + 1,
                            "Experiment": f"Size_{size1}_{size2}",
                            "TimeTrack_Size": size1,
                            "Dataset2_3_Size": size2,
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Samples": len(series),
                            "Train Time (s)": round(training_time, 4),
                            "Memory Used (MB)": round(mem_used, 2),
                            "Max CPU (%)": round(max_cpu, 2),
                            "Avg CPU (%)": round(avg_cpu, 2),
                            "MAE": mae,
                            "MSE": mse,
                            "MAPE": mape
                        })
                        
                    except Exception as e:
                        print(f"Error training {model_name} on {dataset_name}: {str(e)}")
                        continue
        
        # Convert current run results to DataFrame
        df_current_run = pd.DataFrame(run_results)
        
        # Append to existing results
        if df_all_runs.empty:
            df_all_runs = df_current_run
        else:
            df_all_runs = pd.concat([df_all_runs, df_current_run], ignore_index=True)
        
        # Save updated results after each run
        df_all_runs.to_csv("avg.csv", index=False)
        print(f"Results saved after run {run + 1}")
        
        # Calculate and save average results after each run
        avg_columns = ['Train Time (s)', 'Memory Used (MB)', 'Max CPU (%)', 'Avg CPU (%)', 'MAE', 'MSE', 'MAPE']
        groupby_columns = ['Experiment', 'TimeTrack_Size', 'Dataset2_3_Size', 'Dataset', 'Model', 'Samples']
        
        df_avg_results = df_all_runs.groupby(groupby_columns)[avg_columns].mean().reset_index()
        df_avg_results = df_avg_results.round({
            'Train Time (s)': 4,
            'Memory Used (MB)': 2,
            'Max CPU (%)': 2,
            'Avg CPU (%)': 2,
            'MAE': 6,
            'MSE': 6,
            'MAPE': 4
        })
        
        # Save average results
        df_avg_results.to_csv("average_computational_complexity_results.csv", index=False)
        
        # Calculate standard deviations
        df_std_results = df_all_runs.groupby(groupby_columns)[avg_columns].std().reset_index()
        df_std_results = df_std_results.round({
            'Train Time (s)': 4,
            'Memory Used (MB)': 2,
            'Max CPU (%)': 2,
            'Avg CPU (%)': 2,
            'MAE': 6,
            'MSE': 6,
            'MAPE': 4
        })
        df_std_results.to_csv("std_dev_computational_complexity_results.csv", index=False)
        
        print(f"Average results updated after run {run + 1}")
        print(f"Progress: {run + 1}/{NUM_RUNS} runs completed ({((run + 1) / NUM_RUNS) * 100:.1f}%)")

except KeyboardInterrupt:
    print("\nExperiment interrupted by user!")
    print("All results up to this point have been saved.")
except Exception as e:
    print(f"\nUnexpected error: {e}")
    print("All results up to this point have been saved.")

# Final summary
print(f"\n{'='*60}")
print("EXPERIMENT COMPLETE (OR INTERRUPTED)")
print(f"{'='*60}")
print(f"Total runs completed: {len(df_all_runs['Run'].unique()) if not df_all_runs.empty else 0}")
print(f"Individual run results saved to: avg.csv")
print(f"Average results saved to: average_computational_complexity_results.csv")
print(f"Standard deviations saved to: std_dev_computational_complexity_results.csv")

if not df_all_runs.empty:
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total experiments per run: {len(data_sizes)} data sizes × {len(data_dict)} datasets × {len(model_builders)} models")
    print(f"Total runs: {len(df_all_runs['Run'].unique())}")
    print(f"Average training time across all experiments: {df_all_runs['Train Time (s)'].mean():.2f} seconds")
    print(f"Average memory usage across all experiments: {df_all_runs['Memory Used (MB)'].mean():.2f} MB")
    print(f"Average CPU usage across all experiments: {df_all_runs['Avg CPU (%)'].mean():.2f}%")