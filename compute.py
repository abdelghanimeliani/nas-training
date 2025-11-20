import pandas as pd
import numpy as np
import time
import psutil, os, threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Conv1D, Flatten, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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
    "TCN": build_tcn,
    "Transformer": build_transformer,
    "MLP": build_mlp
}

# ============================================================
# 4. EXPERIMENT WITH DIFFERENT DATA SIZES
# ============================================================

# Define different data sizes to test (keeping 6300/945 ratio)
data_sizes = [
    (630, 95),      # 10% of original
    (1260, 189),    # 20% of original
    (1890, 284),    # 30% of original
    (2520, 378),    # 40% of original
    (3150, 473),    # 50% of original
    (3780, 567),    # 60% of original
    (4410, 662),    # 70% of original
    (5040, 756),    # 80% of original
    (5670, 851),    # 90% of original
    (6300, 945),    # 100% of original
    (12600, 1890),  # 200% of original
    (18900, 2835),  # 300% of original
    (25200, 3780),  # 400% of original
    (31500, 4725),  # 500% of original
    (37800, 5670),  # 600% of original (close to max 40000)
]

all_results = []

WINDOW = 10
EPOCHS = 25
BATCH = 16
EPSILON = 1e-8  # for safe MAPE calculation

# Load all available data first
full_data_dict = load_data()

for size_idx, (size1, size2) in enumerate(data_sizes):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {size_idx + 1}/{len(data_sizes)}")
    print(f"Data sizes: TimeTrack={size1}, Dataset2/3={size2}")
    print(f"{'='*60}")
    
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

                all_results.append({
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

# ============================================================
# 5. SAVE RESULTS
# ============================================================

df_all_results = pd.DataFrame(all_results)
df_all_results.to_csv("computational_complexity_scaling_results.csv", index=False)

print("\nSaved results to computational_complexity_scaling_results.csv")

# ============================================================
# 6. PLOT SCALING ANALYSIS
# ============================================================

# Set style for clean plots
plt.style.use('default')
sns.set_palette("husl")

# Create scaling analysis plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Computational Scaling Analysis with Different Data Sizes', fontsize=16, fontweight='bold')

# Plot 1: Training Time vs Data Size (TimeTrack dataset)
time_track_data = df_all_results[df_all_results['Dataset'] == 'TimeTrack']
for model in model_builders.keys():
    model_data = time_track_data[time_track_data['Model'] == model]
    if not model_data.empty:
        axes[0, 0].plot(model_data['TimeTrack_Size'], model_data['Train Time (s)'], 
                       marker='o', label=model, linewidth=2)
axes[0, 0].set_title('Training Time vs Data Size (TimeTrack)')
axes[0, 0].set_xlabel('Data Size')
axes[0, 0].set_ylabel('Training Time (seconds)')
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Memory Usage vs Data Size (TimeTrack dataset)
for model in model_builders.keys():
    model_data = time_track_data[time_track_data['Model'] == model]
    if not model_data.empty:
        axes[0, 1].plot(model_data['TimeTrack_Size'], model_data['Memory Used (MB)'], 
                       marker='s', label=model, linewidth=2)
axes[0, 1].set_title('Memory Usage vs Data Size (TimeTrack)')
axes[0, 1].set_xlabel('Data Size')
axes[0, 1].set_ylabel('Memory Used (MB)')
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: CPU Usage vs Data Size (TimeTrack dataset)
for model in model_builders.keys():
    model_data = time_track_data[time_track_data['Model'] == model]
    if not model_data.empty:
        axes[0, 2].plot(model_data['TimeTrack_Size'], model_data['Avg CPU (%)'], 
                       marker='^', label=model, linewidth=2)
axes[0, 2].set_title('CPU Usage vs Data Size (TimeTrack)')
axes[0, 2].set_xlabel('Data Size')
axes[0, 2].set_ylabel('Avg CPU Usage (%)')
axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Training Time vs Data Size (Dataset2)
dataset2_data = df_all_results[df_all_results['Dataset'] == 'Dataset2']
for model in model_builders.keys():
    model_data = dataset2_data[dataset2_data['Model'] == model]
    if not model_data.empty:
        axes[1, 0].plot(model_data['Dataset2_3_Size'], model_data['Train Time (s)'], 
                       marker='o', label=model, linewidth=2)
axes[1, 0].set_title('Training Time vs Data Size (Dataset2)')
axes[1, 0].set_xlabel('Data Size')
axes[1, 0].set_ylabel('Training Time (seconds)')
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Memory Usage vs Data Size (Dataset2)
for model in model_builders.keys():
    model_data = dataset2_data[dataset2_data['Model'] == model]
    if not model_data.empty:
        axes[1, 1].plot(model_data['Dataset2_3_Size'], model_data['Memory Used (MB)'], 
                       marker='s', label=model, linewidth=2)
axes[1, 1].set_title('Memory Usage vs Data Size (Dataset2)')
axes[1, 1].set_xlabel('Data Size')
axes[1, 1].set_ylabel('Memory Used (MB)')
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: CPU Usage vs Data Size (Dataset2)
for model in model_builders.keys():
    model_data = dataset2_data[dataset2_data['Model'] == model]
    if not model_data.empty:
        axes[1, 2].plot(model_data['Dataset2_3_Size'], model_data['Avg CPU (%)'], 
                       marker='^', label=model, linewidth=2)
axes[1, 2].set_title('CPU Usage vs Data Size (Dataset2)')
axes[1, 2].set_xlabel('Data Size')
axes[1, 2].set_ylabel('Avg CPU Usage (%)')
axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 7. SUMMARY STATISTICS
# ============================================================

print("\n" + "="*60)
print("SCALING ANALYSIS SUMMARY")
print("="*60)

# Calculate scaling factors
scaling_stats = []
for size1, size2 in data_sizes:
    size_data = df_all_results[df_all_results['TimeTrack_Size'] == size1]
    if not size_data.empty:
        avg_time = size_data['Train Time (s)'].mean()
        avg_memory = size_data['Memory Used (MB)'].mean()
        avg_cpu = size_data['Avg CPU (%)'].mean()
        scaling_stats.append({
            'TimeTrack_Size': size1,
            'Dataset2_3_Size': size2,
            'Avg_Train_Time': avg_time,
            'Avg_Memory_Used': avg_memory,
            'Avg_CPU_Usage': avg_cpu
        })

scaling_df = pd.DataFrame(scaling_stats)
print("\nAverage Metrics by Data Size:")
print(scaling_df.round(3))

# Calculate scaling efficiency
if len(scaling_df) > 1:
    base_size = scaling_df.iloc[0]['TimeTrack_Size']
    base_time = scaling_df.iloc[0]['Avg_Train_Time']
    
    print(f"\nScaling Efficiency (relative to {base_size} samples):")
    for idx, row in scaling_df.iterrows():
        size_ratio = row['TimeTrack_Size'] / base_size
        time_ratio = row['Avg_Train_Time'] / base_time
        efficiency = size_ratio / time_ratio if time_ratio > 0 else 0
        print(f"Size {row['TimeTrack_Size']}: {efficiency:.3f}x efficiency")

print("\nAll results saved to:")
print("- computational_complexity_scaling_results.csv")
print("- scaling_analysis_comprehensive.png")