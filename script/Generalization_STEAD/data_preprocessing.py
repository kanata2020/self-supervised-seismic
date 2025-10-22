import pandas as pd
import h5py
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Get the directory of the current script
current_dir = Path(__file__).parent

# File paths for earthquake and noise data
earthquake_file_name = current_dir / "earthquake" / "chunk.hdf5"#choose the chunk you want
earthquake_csv_file = current_dir / "earthquake" / "chunk.csv"
noise_file_name = current_dir / "noise" / "chunk.hdf5"#choose the chunk you want
noise_csv_file = current_dir / "noise" / "chunk.csv"

# read csv data
df_earthquake = pd.read_csv(earthquake_csv_file)
df_noise = pd.read_csv(noise_csv_file)

# Filter earthquake data
df_earthquake = df_earthquake[(df_earthquake.trace_category == 'earthquake_local') & 
                               (df_earthquake.source_distance_km <= 21) & 
                               (df_earthquake.source_magnitude > 0.2)]
print(f'Total earthquake events: {len(df_earthquake)}')

# Filter noise data
df_noise = df_noise[(df_noise.trace_category == 'noise')] #& (df_noise.receiver_code == 'PHOB')]
print(f'Total noise events: {len(df_noise)}')

# # Randomly select 2000 events from earthquake and noise data
earthquake_sample = df_earthquake.sample(n=2000, random_state=42)
noise_sample = df_noise.sample(n=2000, random_state=42)

# Get trace names for selected earthquake and noise events
earthquake_trace_names = earthquake_sample['trace_name'].to_list()
noise_trace_names = noise_sample['trace_name'].to_list()

# Initialize lists to store waveform data
earthquake_data = []
noise_data = []

# read earthquake waveform data
with h5py.File(earthquake_file_name, 'r') as dtfl:
    for trace_name in earthquake_trace_names:
        dataset = dtfl.get('data/' + str(trace_name))
        data = np.array(dataset)
        earthquake_data.append(data)

# read noise waveform data
with h5py.File(noise_file_name, 'r') as dtfl:
    for trace_name in noise_trace_names:
        dataset = dtfl.get('data/' + str(trace_name))
        data = np.array(dataset)
        noise_data.append(data)

# transfer waveform to numpy array
earthquake_data = np.array(earthquake_data)
noise_data = np.array(noise_data)

# new time line
original_time = np.linspace(0, 1, earthquake_data.shape[1])  # [0, 1] 区间，代表 6000 个样本
new_time = np.linspace(0, 1, 3750)  # [0, 1] 区间，代表 3750 个样本

# Perform interpolation for each earthquake and noise event and each channel
earthquake_data_interpolated = np.zeros((earthquake_data.shape[0], 3750, earthquake_data.shape[2]))

for i in range(earthquake_data.shape[0]):  
    for j in range(earthquake_data.shape[2]): 
        interpolator = interp1d(original_time, earthquake_data[i, :, j], kind='linear', fill_value="extrapolate")
        earthquake_data_interpolated[i, :, j] = interpolator(new_time)




noise_data_interpolated = np.zeros((noise_data.shape[0], 3750, noise_data.shape[2]))

for i in range(noise_data.shape[0]):  
    for j in range(noise_data.shape[2]):  
        interpolator = interp1d(original_time, noise_data[i, :, j], kind='linear', fill_value="extrapolate")
        noise_data_interpolated[i, :, j] = interpolator(new_time)



# save as .npy 
np.save('earthquake_data.npy', earthquake_data)
np.save('noise_data.npy', noise_data)

import numpy as np

# # Combine earthquake and noise data
x_gen_test = np.concatenate((earthquake_data_interpolated, noise_data_interpolated), axis=0)

# # Combine earthquake and noise label
y_gen_test = np.concatenate((np.zeros(earthquake_data_interpolated.shape[0]), np.ones(noise_data_interpolated.shape[0])), axis=0)

# print results and shape
print(f'x_gen_test shape: {x_gen_test.shape}')  # [400, 3750, 3]
print(f'y_gen_test shape: {y_gen_test.shape}')  # [400,]

fig = plt.figure()
plt.plot(x_gen_test[6,:,2])  
plt.show()  

#save data for generalization test
np.save('x_gen_test.npy', x_gen_test)
np.save('y_gen_test.npy', y_gen_test)