import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import skew, kurtosis
import matplotlib.patches as mpatches

#Loading trained model and scaler
model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

#Data columns
TIME_COL = "Time (s)"
AX_COL   = "Acceleration x (m/s^2)"
AY_COL   = "Acceleration y (m/s^2)"
AZ_COL   = "Acceleration z (m/s^2)"
ABS_COL  = "Absolute acceleration (m/s^2)"

feature_cols = [AX_COL, AY_COL, AZ_COL, ABS_COL]

#Defining helper functions
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ffill(inplace=True)
    return df

def smooth_signals(df: pd.DataFrame, window_size=10) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
    return df

def extract_window_features(window: pd.DataFrame) -> dict:
    feats = {}
    for col in feature_cols:
        if col not in window.columns:
            continue
        col_data = window[col]
        feats[f"{col}_mean"]     = col_data.mean()
        feats[f"{col}_std"]      = col_data.std()
        feats[f"{col}_min"]      = col_data.min()
        feats[f"{col}_max"]      = col_data.max()
        feats[f"{col}_skew"]     = skew(col_data)
        feats[f"{col}_kurtosis"] = kurtosis(col_data)
        feats[f"{col}_range"]    = col_data.max() - col_data.min()
        feats[f"{col}_median"]   = col_data.median()
        feats[f"{col}_var"]      = col_data.var()
        feats[f"{col}_mad"]      = np.mean(np.abs(col_data - col_data.mean()))
    return feats

def segment_and_extract(df: pd.DataFrame, window_sec=5):
    dt = df[TIME_COL].diff().dropna()
    if dt.empty or dt.median() == 0:
        return pd.DataFrame(), []

    sampling_rate = 1 / dt.median()
    samples_per_window = int(window_sec * sampling_rate)

    X_list = []
    window_times = []
    start_idx = 0
    while start_idx + samples_per_window <= len(df):
        window_data = df.iloc[start_idx : start_idx + samples_per_window]
        feats = extract_window_features(window_data)
        X_list.append(feats)

        # Record the time span
        t0 = window_data[TIME_COL].iloc[0]
        t1 = window_data[TIME_COL].iloc[-1]
        window_times.append((t0, t1))

        start_idx += samples_per_window

    X_df = pd.DataFrame(X_list)
    #filling any potential gaps
    if not X_df.empty:
        X_df.fillna(X_df.mean(), inplace=True)
    return X_df, window_times

#Classify Data and show in GUI
def classify_csv(file_path: str, output_text: tk.Text, plot_frame: tk.Frame):
    output_text.delete("1.0", tk.END)

    #Load CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
        return

    if TIME_COL not in df.columns:
        messagebox.showerror("Error", f"CSV missing '{TIME_COL}' column.")
        return

    #fill + smooth
    df = fill_missing_values(df)
    df = smooth_signals(df, window_size=10)

    # segment + extract
    X_df, window_times = segment_and_extract(df, window_sec=5)
    if X_df.empty:
        messagebox.showwarning("No windows", "No complete 5-second windows found.")
        return

    X_scaled = scaler.transform(X_df)

    #predict
    preds = model.predict(X_scaled)

    #Display textual results
    output_text.insert(tk.END, f"File: {file_path}\n\n")
    output_text.insert(tk.END, "WINDOW_ID   TIME_RANGE        PREDICTION\n")
    for i, (start_t, end_t) in enumerate(window_times):
        label = preds[i]
        window_id = i + 1
        output_text.insert(
            tk.END,
            f"{window_id:>9}   [{start_t:.2f} - {end_t:.2f}]   {label}\n"
        )

    #Plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    line_x, = ax.plot(df[TIME_COL], df[AX_COL], label="Accel X")
    line_y, = ax.plot(df[TIME_COL], df[AY_COL], label="Accel Y")
    line_z, = ax.plot(df[TIME_COL], df[AZ_COL], label="Accel Z")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("Accelerations with Predicted Windows")
    ax.grid(True)

    y_min, y_max = ax.get_ylim()
    for i, (start_t, end_t) in enumerate(window_times):
        label = preds[i]
        color = "green" if label == "walking" else "red"
        ax.add_patch(Rectangle(
            (start_t, y_min),
            end_t - start_t,
            y_max - y_min,
            color=color,
            alpha=0.1
        ))

    #Build one legend that includes X, Y, Z plus walking/jumping patches
    walk_patch = mpatches.Patch(color='green', alpha=0.1, label='Walking Window')
    jump_patch = mpatches.Patch(color='red', alpha=0.1, label='Jumping Window')

    #Current line handles + labels
    lines, labels = ax.get_legend_handles_labels()

    lines += [walk_patch, jump_patch]
    labels += ["Walking Window", "Jumping Window"]

    ax.legend(lines, labels, loc="upper right")

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#MAIN GUI
def open_file_dialog(output_text: tk.Text, plot_frame: tk.Frame):
    file_path = filedialog.askopenfilename(
        title="Select Accelerometer CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if file_path:
        classify_csv(file_path, output_text, plot_frame)

def main():
    root = tk.Tk()
    root.title("Walking vs Jumping Classifier")

    #top frame for buttons
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    #Scrolled text for output
    output_text = scrolledtext.ScrolledText(root, width=70, height=22, font=("Courier", 10))
    output_text.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    #Frame for plot
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    browse_btn = tk.Button(top_frame, text="Browse CSV", command=lambda: open_file_dialog(output_text, plot_frame))
    browse_btn.pack(side=tk.LEFT, padx=10, pady=10)

    quit_btn = tk.Button(top_frame, text="Quit", command=root.destroy)
    quit_btn.pack(side=tk.RIGHT, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
