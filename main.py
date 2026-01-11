from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    # Ensure imports work when running from outside this directory.
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = SCRIPT_DIR
TRAIN_DIR = SCRIPT_DIR / "train"
TEST_DIR = SCRIPT_DIR / "tests"

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------
# STEP 1 – DATA COLLECTION
# ------------------------------------
import pandas as pd
from processing import extract_features, load_trimmed

#Loading each CSV file
julien_walking_1 = pd.read_csv(TEST_DIR / "julien_walking_1.csv")
julien_walking_2 = pd.read_csv(TRAIN_DIR / "julien_walking_2.csv")
julien_walking_3 = pd.read_csv(TRAIN_DIR / "julien_walking_3.csv")
julien_jumping_1 = pd.read_csv(TEST_DIR / "julien_jumping_1.csv")
julien_jumping_2 = pd.read_csv(TRAIN_DIR / "julien_jumping_2.csv")
julien_jumping_3 = pd.read_csv(TRAIN_DIR / "julien_jumping_3.csv")

#Function to add activity and subject label to DataFrame
def label_activity_subject(df, activity, subject):
    df["Activity"] = activity
    df["Subject"] = subject

label_activity_subject(julien_walking_1, "walking", "julien")
label_activity_subject(julien_walking_2, "walking", "julien")
label_activity_subject(julien_walking_3, "walking", "julien")
label_activity_subject(julien_jumping_1, "jumping", "julien")
label_activity_subject(julien_jumping_2, "jumping", "julien")
label_activity_subject(julien_jumping_3, "jumping", "julien")

#Combine all labeled datasets
combined_data = pd.concat([
    julien_walking_1, julien_walking_2, julien_walking_3,
    julien_jumping_1, julien_jumping_2, julien_jumping_3
], ignore_index=True)

#Hold out walking_1 and jumping_1 for testing only.
train_data = pd.concat([
    julien_walking_2, julien_walking_3,
    julien_jumping_2, julien_jumping_3
], ignore_index=True)
test_data = pd.concat([
    julien_walking_1, julien_jumping_1
], ignore_index=True)

print("Step 1 Complete: Data loaded, labeled, and split into train/test folders.")

# ---------------------------------------
# STEP 2 – DATA STORAGE
# --------------------------------------
import h5py
import numpy as np

#Open HDF5 file and create raw group
hdf5_file = h5py.File(DATA_DIR / "activity_data.h5", "w")
raw_group = hdf5_file.create_group("raw")

#Filtering combined data by subject
julien_data = combined_data[combined_data["Subject"] == "julien"]


#Function to store subject data into HDF5
def store_subject_data(h5_group, subject_name, data):
    numeric = data.select_dtypes(include=[np.number])
    strings = data.select_dtypes(include=["object"]).astype("S")
    subj_group = h5_group.create_group(subject_name)
    subj_group.create_dataset("numeric_data", data=numeric.values)
    subj_group.create_dataset("string_data", data=strings.values)
    subj_group.attrs["numeric_columns"] = list(numeric.columns)
    subj_group.attrs["string_columns"] = list(strings.columns)

#Store data for each subject
store_subject_data(raw_group, "julien", julien_data)

hdf5_file.close()
print("Step 2 Complete: Raw data stored in HDF5.")

# ------------------------------------
# STEP 3 – VISUALIZATION
# ------------------------------------

#Plot acceleration data on given axis
def plot_accel(ax, df, title):
    ax.plot(df["Time (s)"], df["Acceleration x (m/s^2)"], label="X-axis")
    ax.plot(df["Time (s)"], df["Acceleration y (m/s^2)"], label="Y-axis")
    ax.plot(df["Time (s)"], df["Acceleration z (m/s^2)"], label="Z-axis")
    ax.set_title(title)
    ax.set_ylabel("Accel (m/s²)")
    ax.grid(True)
    ax.legend()

#Visualization sample
julien_walk = load_trimmed(TEST_DIR / "julien_walking_1.csv")
julien_jump = load_trimmed(TEST_DIR / "julien_jumping_1.csv")


#Create subplots for each sample plot
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
plot_accel(axs[0], julien_walk, "Julien – Walking")
plot_accel(axs[1], julien_jump, "Julien – Jumping")

axs[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show() #Display plots

# -------------------------------------
# STEP 4 – PREPROCESSING
# --------------------------------------

#Fill missing values using forward filling
preprocessed_train = train_data.copy()
preprocessed_train.ffill(inplace=True)

preprocessed_test = test_data.copy()
preprocessed_test.ffill(inplace=True)

#Apply moving average
cols_to_smooth = [
    "Acceleration x (m/s^2)",
    "Acceleration y (m/s^2)",
    "Acceleration z (m/s^2)",
    "Absolute acceleration (m/s^2)"
]
for c in cols_to_smooth:
    if c in preprocessed_train.columns:
        preprocessed_train[c] = preprocessed_train[c].rolling(window=10, center=True, min_periods=1).mean()
    if c in preprocessed_test.columns:
        preprocessed_test[c] = preprocessed_test[c].rolling(window=10, center=True, min_periods=1).mean()

#plot raw vs. smoothed of first 300 julien samples
julien_raw = train_data[train_data["Subject"] == "julien"].reset_index(drop=True)
julien_pre = preprocessed_train[preprocessed_train["Subject"] == "julien"].reset_index(drop=True)

plt.figure(figsize=(10, 4))
plt.plot(
    julien_raw["Time (s)"][:300],
    julien_raw["Absolute acceleration (m/s^2)"][:300],
    label="Raw Abs Accel", alpha=0.5
)
plt.plot(
    julien_pre["Time (s)"][:300],
    julien_pre["Absolute acceleration (m/s^2)"][:300],
    label="Smoothed Abs Accel", linewidth=2
)
plt.title("Absolute Acceleration: Raw vs Smoothed (Julien, first 300 samples)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Step 4 Complete: Missing values handled, noise reduced with moving average.")

# ----------------------------------------
# STEP 5 – FEATURE EXTRACTION + NORMALIZATION
# ----------------------------------------
from sklearn.preprocessing import StandardScaler

train_data_prepped = preprocessed_train.copy()
test_data_prepped = preprocessed_test.copy()

#Determine sampling rate from time differences
time_diffs = train_data_prepped["Time (s)"].diff().dropna()
sampling_rate = 1 / time_diffs.median()
samples_per_window = int(5 * sampling_rate)

def build_feature_frame(dataframe, window_size):
    x_list = []
    y_list = []
    # Group by subject, activity
    for (_, act), group_df in dataframe.groupby(["Subject", "Activity"]):
        group_df = group_df.reset_index(drop=True)
        for start in range(0, len(group_df) - window_size, window_size):
            window = group_df.iloc[start:start + window_size]
            if len(window) == window_size:
                feats = extract_features(window)
                x_list.append(feats)
                y_list.append(act)
    x_df = pd.DataFrame(x_list)
    y_series = pd.Series(y_list, name="Label")
    return x_df, y_series

X_train_df, y_train_series = build_feature_frame(train_data_prepped, samples_per_window)
X_test_df, y_test_series = build_feature_frame(test_data_prepped, samples_per_window)

train_means = X_train_df.mean()
X_train_df.fillna(train_means, inplace=True) #handle missing feature values
X_test_df = X_test_df.reindex(columns=X_train_df.columns)
X_test_df.fillna(train_means, inplace=True)

#Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_df.columns)
X_train_scaled_df["Label"] = y_train_series.values

X_test_scaled = scaler.transform(X_test_df)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train_df.columns)
X_test_scaled_df["Label"] = y_test_series.values

print("Step 5 Complete: Extracted and normalized features (train and held-out test).")

# ----------------------------------------
# STEP 6 – CLASSIFIER TRAINING + EVALUATION
# ----------------------------------------
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

#Preparing features for training/testing
X_train = X_train_scaled_df.drop("Label", axis=1)
y_train = X_train_scaled_df["Label"]
X_test = X_test_scaled_df.drop("Label", axis=1)
y_test = X_test_scaled_df["Label"]

#Training logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Compute model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy = {accuracy * 100:.2f}%")

#Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

#Generate learning curve for further evaluation
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=LogisticRegression(max_iter=1000),
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, valid_mean, 'o-', label='Validation Accuracy')
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1)
plt.title("Learning Curve (Logistic Regression)")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

print("Step 6 Complete: Model trained, tested, and learning curves generated.")

#Save trained model and scaler for future use (GUI)
joblib.dump(model, DATA_DIR / "logreg_model.pkl")
joblib.dump(scaler, DATA_DIR / "scaler.pkl")
print("Saved logistic regression model to 'logreg_model.pkl' and scaler to 'scaler.pkl'.")

# --------------------------------------
# STEP 7 – STORE PREPROCESSED + SEGMENTED INTO HDF5
# -------------------------------------
with h5py.File(DATA_DIR / "activity_data.h5", "a") as h5f:
    #Storing preprocessed data per subject
    preprocessed_group = h5f.require_group("preprocessed")
    for subject in preprocessed_train["Subject"].unique():
        subject_data = preprocessed_train[preprocessed_train["Subject"] == subject]
        numeric = subject_data.select_dtypes(include=[np.number])
        strings = subject_data.select_dtypes(include=["object"]).astype("S")

        subj_group = preprocessed_group.create_group(subject)
        subj_group.create_dataset("numeric_data", data=numeric.values)
        subj_group.create_dataset("string_data", data=strings.values)
        subj_group.attrs["numeric_columns"] = list(numeric.columns)
        subj_group.attrs["string_columns"] = list(strings.columns)

    print("Preprocessed data stored into HDF5.")

    segmented_group = h5f.require_group("segmented")
    segmented_group.create_dataset("train", data=X_train.values)
    segmented_group.create_dataset("train_labels", data=np.array(y_train).astype("S"))
    segmented_group.create_dataset("test", data=X_test.values)
    segmented_group.create_dataset("test_labels", data=np.array(y_test).astype("S"))

    print("Segmented train/test data stored into HDF5.")
