import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Filter short sequences from the numpy array
def filter_short_sequences(df, group_col, min_len):
    
    counts = df.groupby(group_col).size()
    
    valid_ids = counts[counts >= min_len].index
    
    df_filtered = df[df[group_col].isin(valid_ids)].reset_index(drop=True)
    return df_filtered

def cap_long_seq(df, group_col, max_len=100, keep='start'):
    capped_rows = []

    for obj_id, group in df.groupby(group_col):
        group = group.sort_values('Time (MJD)')

        if len(group) > max_len:
            if keep == 'start':
                group = group.iloc[:max_len]
            elif keep == 'end':
                group = group.iloc[-max_len:]
            else:
                raise ValueError("keep must be 'start' or 'end'")
        
        capped_rows.append(group)

    df_capped = pd.concat(capped_rows, ignore_index=True)
    return df_capped

def split_long_seq(df, group_col, max_len=100):
    new_rows = []

    for obj_id, group in df.groupby(group_col):
        group = group.sort_values('Time (MJD)')
        n_steps = len(group)
        if n_steps <= max_len:
            continue  # skip short sequences

        n_splits = (n_steps + max_len - 1) // max_len
        for i in range(n_splits):
            start = i * max_len
            end = start + max_len
            sub_group = group.iloc[start:end].copy()
            sub_group[group_col] = f"new_{obj_id}_{i+1}"
            new_rows.append(sub_group)

    if new_rows:
        df_new_only = pd.concat(new_rows, ignore_index=True)
    else:
        df_new_only = pd.DataFrame(columns=df.columns)

    # Include sequences that were already <= max_len
    df_short = df.groupby(group_col).filter(lambda x: len(x) <= max_len)
    if not df_new_only.empty:
        df_final = pd.concat([df_short, df_new_only], ignore_index=True)
    else:
        df_final = df_short.reset_index(drop=True)

    return df_final

def prepare_sequence(df_train, df_test, group_col, feature_cols, target_col, max_seq_len=100):
    def process(df, include_target=True):
        X_list, y_list = [], []
        grouped = df.groupby(group_col)
        for _, group in grouped:
            group = group.sort_values('Time (MJD)')
            X_seq = group[feature_cols].values
            X_list.append(X_seq)
            if include_target:
                y_list.append(group[target_col].iloc[0])
        X_padded = pad_sequences(X_list, maxlen=max_seq_len, padding='post', dtype='float32')
        y_array = np.array(y_list) if include_target else None
        return X_padded, y_array

    X_train, y_train = process(df_train, include_target=True)
    X_test, _ = process(df_test, include_target=False)
    return X_train, y_train, X_test

def GP_aug(X, y, n_aug=1, random_state=42, balance=False):
    rng = np.random.default_rng(random_state)
    N, T, C = X.shape
    X_new = []
    y_new = []

    # fixed kernel: length_scale and noise_level are not optimized
    kernel = RBF(length_scale=0.15, length_scale_bounds="fixed") + \
             WhiteKernel(noise_level=1e-3, noise_level_bounds="fixed")

    for i in range(N):
        if balance and (y[i] !=1):
            continue
        x_orig = X[i]
        for _ in range(n_aug):
            x_syn = np.zeros_like(x_orig)
            for ch in range(C):
                flux = x_orig[:, ch]
                valid_idx = np.where(flux != 0)[0]
                if len(valid_idx) < 2:
                    continue
                t_valid = (valid_idx / T).reshape(-1, 1)
                y_valid = flux[valid_idx]
                gp = GaussianProcessRegressor(kernel=kernel, random_state=random_state, n_restarts_optimizer=0)
                gp.fit(t_valid, y_valid)
                pred, std = gp.predict(t_valid, return_std=True)
                noise = rng.normal(0, 0.15 * std)
                x_syn[valid_idx, ch] = pred + noise
            X_new.append(x_syn)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

def Noise_aug(X, y, noise_level=0.05, n_aug=1, random_state=42, balance=False):
    rng = np.random.default_rng(random_state)
    X_new = []
    y_new = []

    for i in range(X.shape[0]):
        if balance and (y[i] !=1):
            continue
        for _ in range(n_aug):
            x_orig = X[i].copy()
            mask = x_orig != 0
            x_orig[mask] += rng.normal(0, noise_level * np.abs(x_orig[mask]), size=np.sum(mask))
            X_new.append(x_orig)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

def AvgPair_segments(X, y, n_aug=1, balance=False):
    X_new, y_new = [], []

    for i in range(X.shape[0]):
        if balance and y[i] != 1:
            continue

        x = X[i].astype(np.float32)

        for _ in range(n_aug):
            x_aug = np.zeros_like(x)

            for ch in range(x.shape[1]):
                nz_mask = x[:, ch] != 0
                idx = np.where(nz_mask)[0]

                if len(idx) == 0:
                    continue

                # split into continuous segments
                splits = np.where(np.diff(idx) != 1)[0] + 1
                segments = np.split(idx, splits)

                for seg in segments:
                    if len(seg) < 2:
                        continue

                    for k in range(1, len(seg)):
                        t_prev = seg[k - 1]
                        t = seg[k]
                        x_aug[t, ch] = 0.5 * (x[t_prev, ch] + x[t, ch])

            X_new.append(x_aug)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

def Noise_aug_with_shift(X, y, noise_level=0.05, shift_range=(-10, 15), n_aug=1, random_state=42, balance=False):
    rng = np.random.default_rng(random_state)
    X_new = []
    y_new = []

    for i in range(X.shape[0]):
        if balance and (y[i] != 1):
            continue
        for _ in range(n_aug):
            x_orig = X[i].copy()
            mask = x_orig != 0
            x_orig[mask] += rng.normal(0, noise_level * np.abs(x_orig[mask]), size=np.sum(mask))
            flux_shift = rng.uniform(*shift_range)
            x_orig[mask] += flux_shift
            X_new.append(x_orig)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

def Scale_aug(X, y, scale_range=(0.7, 1.5), n_aug=1, random_state=42, balance=False):
    rng = np.random.default_rng(random_state)
    X_new = []
    y_new = []

    for i in range(X.shape[0]):
        if balance and (y[i] != 1):
            continue
        for _ in range(n_aug):
            x_orig = X[i].copy()
            alpha = rng.uniform(*scale_range)
            mask = x_orig != 0
            x_orig[mask] *= alpha
            X_new.append(x_orig)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

def ChannelDrop_aug(X, y, n_aug=1, random_state=42, balance=False, max_drop=1):
    rng = np.random.default_rng(random_state)
    X_new = []
    y_new = []
    n_channels = (X.shape[2])//2
    for i in range(X.shape[0]):
        if balance and y[i] != 1:
            continue

        x = X[i]

        for _ in range(n_aug):
            x_aug = x.copy()

            n_drop = rng.integers(1, min(max_drop, n_channels) + 1)
            drop_channels = rng.choice(n_channels, size=n_drop, replace=False)
            new_drop = []
            for i in drop_channels:
                new_drop+= [int(i*2), int((i*2)+1)]
            x_aug[:, new_drop] = 0.0

            X_new.append(x_aug)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

import numpy as np

def TimeMask_aug(X, y, n_aug=1, max_frac=0.1, random_state=42, balance=False):
    rng = np.random.default_rng(random_state)
    X_new, y_new = [], []

    T = X.shape[1]
    n_total_channels = X.shape[2]
    n_pairs = n_total_channels // 2

    for i in range(X.shape[0]):
        if balance and y[i] != 1:
            continue

        for _ in range(n_aug):
            x_aug = X[i].copy()
            for p in range(n_pairs):
                max_L = int(T * max_frac)
                if max_L < 1: max_L = 1
                L = rng.integers(1, max_L + 1)
                start = rng.integers(0, T - L + 1)
                
                col_indices = [p * 2, (p * 2) + 1]
                
                x_aug[start:start+L, col_indices] = 0
            
            X_new.append(x_aug)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)

def TimeShift_aug(X, y, n_aug=1, max_shift=10, random_state=42, balance=False):
    rng = np.random.default_rng(random_state)
    X_new, y_new = [], []

    for i in range(X.shape[0]):
        if balance and y[i] != 1:
            continue

        for _ in range(n_aug):
            shift = rng.integers(-max_shift, max_shift + 1)
            x = X[i].copy()

            if shift > 0:
                # shift forward (delay) → pad beginning with zeros
                x_shifted = np.zeros_like(x)
                x_shifted[shift:] = x[:-shift]

            elif shift < 0:
                # shift backward (early start) → pad end with zeros
                x_shifted = np.zeros_like(x)
                x_shifted[:shift] = x[-shift:]

            else:
                x_shifted = x

            X_new.append(x_shifted)
            y_new.append(y[i])

    return np.stack(X_new), np.array(y_new)


