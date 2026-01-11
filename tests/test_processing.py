from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from processing import extract_features, load_trimmed, standardize_columns, FEATURE_COLS


def test_standardize_columns_maps_headers():
    df = pd.DataFrame(columns=[
        "Time(s)",
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
    ])
    standardized = standardize_columns(df)
    assert standardized.columns.tolist() == [
        "Time (s)",
        "Acceleration x (m/s^2)",
        "Acceleration y (m/s^2)",
        "Acceleration z (m/s^2)",
    ]


def test_load_trimmed_filters_by_time():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "sample.csv"
        df = pd.DataFrame(
            {
                "Time": [0, 5, 10, 15],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [1.0, 1.5, 2.5, 3.5],
                "z": [0.2, 0.4, 0.6, 0.8],
            }
        )
        df.to_csv(csv_path, index=False)
        trimmed = load_trimmed(csv_path, max_time=10)
        assert trimmed["Time (s)"].max() == 10
        assert len(trimmed) == 3


def test_extract_features_computes_basic_stats():
    window = pd.DataFrame({col: [1.0, 2.0, 3.0, 4.0] for col in FEATURE_COLS})
    feats = extract_features(window)
    assert len(feats) == len(FEATURE_COLS) * 10

    col = FEATURE_COLS[0]
    assert np.isclose(feats[f"{col}_mean"], 2.5)
    assert feats[f"{col}_min"] == 1.0
    assert feats[f"{col}_max"] == 4.0
    assert feats[f"{col}_range"] == 3.0
    assert np.isfinite(feats[f"{col}_skew"])
    assert np.isfinite(feats[f"{col}_kurtosis"])


def run_all_tests():
    module = sys.modules[__name__]
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        func = getattr(module, name)
        if callable(func):
            func()


if __name__ == "__main__":
    run_all_tests()
    print("All tests passed.")
