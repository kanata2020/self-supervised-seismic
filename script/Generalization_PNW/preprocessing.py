import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from scipy.signal import butter, filtfilt

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

def load_pnw_mini(waveform_h5_path, metadata_csv_path,
                  selected_event_types=None, max_per_type=None,
                  target_len=6000):
    """
    Load PNW mini dataset into arrays X, y with fixed trace length.

    waveform_h5_path: path to waveform file (HDF5)
    metadata_csv_path: path to metadata file (CSV)
    selected_event_types: list of event types (strings) to include
    max_per_type: int or dict, limit number of samples per class
    target_len: int, length to pad/crop each trace to

    Returns:
      X: numpy array, shape (N, 3, target_len)
      y: numpy array, integer labels
      label_map: dict(label_name → id)
    """

    def pad_or_crop(trace, target_len=10000):
        """
        pad_or_crop to fixed length: target_len
        """
        T = trace.shape[1]
        if T < target_len:
            return np.pad(trace, ((0,0),(0,target_len - T)), mode='constant')
        else:
            return trace[:, :target_len]



    df = pd.read_csv(metadata_csv_path)
    if selected_event_types is not None:
        df = df[df['source_type'].isin(selected_event_types)]
    df = df.reset_index(drop=True)

    f = h5py.File(waveform_h5_path, "r")

    X_list, y_list = [], []
    label_map = {}
    next_label = 0
    counts = {etype: 0 for etype in selected_event_types}

    print(f"📘 Loading dataset from: {waveform_h5_path}")

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        evt_type = row['source_type']
        trace_name = row['trace_name']

        # check limit
        limit = None
        if isinstance(max_per_type, dict):
            limit = max_per_type.get(evt_type, None)
        elif isinstance(max_per_type, int):
            limit = max_per_type
        if (limit is not None) and (counts[evt_type] >= limit):
            continue

        # parse trace_name safely
        try:
            bucket, rest = trace_name.split('$')
            x_idx = int(rest.split(',')[0])
            data = f[f"/data/{bucket}"][x_idx, :, :]
            data = pad_or_crop(data, target_len)
        except Exception as e:
            print(f"⚠️ Skipping trace {trace_name}: {e}")
            continue

        if evt_type not in label_map:
            label_map[evt_type] = next_label
            next_label += 1

        X_list.append(data)
        y_list.append(label_map[evt_type])
        counts[evt_type] += 1

        # stop early if all categories reached limit
        if all((max_per_type[etype] is None or counts[etype] >= max_per_type[etype]) 
               for etype in selected_event_types):
            break

    f.close()

    if len(X_list) == 0:
        raise ValueError("No valid traces were loaded. Check your CSV and HDF5 paths.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)

    print("✅ Final counts per type:", counts)
    return X, y, label_map


# -------------------- bandpass filter + normlization --------------------
def preprocess_for_encoder(X, y, fs=100, lowcut=2, highcut=49):
    from scipy.signal import butter, filtfilt
    import numpy as np

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=4, Wn=[low, high], btype='band')

    X_filt = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            filtered = filtfilt(b, a, X[i, ch, :])
            max_abs = np.max(np.abs(filtered))
            if max_abs > 0:
                X_filt[i, ch, :] = filtered / max_abs
            else:
                X_filt[i, ch, :] = filtered

    return X_filt, y




if __name__ == "__main__":
    #adjust the path as your data location
    waveform_h5 = "miniPNW_waveforms.hdf5"
    metadata_csv = "miniPNW_metadata.csv"

    selected = ['earthquake', 'explosion', 'surface_event', 'thunder', 'sonic_boom']
    max_per_type = {
        'earthquake': 500,
        'explosion': 500,
        'surface_event': 500,
        'thunder': 94,
        'sonic_boom': 126
    }

    X, y, label_map = load_pnw_mini(
        waveform_h5, metadata_csv,
        selected_event_types=selected,
        max_per_type=max_per_type,
        target_len=15000
    )

    print("✅ Loaded data:", X.shape, "labels:", y.shape)
    print("Label map:", label_map)

    X_norm, y_proc = preprocess_for_encoder(X, y)

    np.save("PNW_dataset/pnw_x_test.npy", X_norm)
    np.save("PNW_dataset/pnw_y_test.npy", y_proc)

    print("💾 Saved pnw_x_test.npy and pnw_y_test.npy")
