import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from pathlib import Path
import h5py
from PIL import Image


CHUNK_SIZE = 33

def save_chunked_data(data, raw_file_path, cleaned_path, sample_index, gzip_compression=False):
    file_name = Path(raw_file_path).name
    for start in range(0, data.shape[0]-CHUNK_SIZE+1, CHUNK_SIZE):
        with h5py.File(cleaned_path / f'{raw_file_path.stem}_s{sample_index}_{start}_{start+CHUNK_SIZE}{raw_file_path.suffix}', 'w') as output_file:
            if gzip_compression:
                output_file.create_dataset('data', data=data[start:start+CHUNK_SIZE], compression="gzip")
            else:
                output_file.create_dataset('data', data=data[start:start+CHUNK_SIZE])

def save_full_data(data, raw_file_path, cleaned_path, sample_index, gzip_compression=False):
    file_name = Path(raw_file_path).name
    with h5py.File(cleaned_path / f'{raw_file_path.stem}_s{sample_index}{raw_file_path.suffix}', 'w') as output_file:
        if gzip_compression:
            output_file.create_dataset('data', data=data, compression="gzip")
        else:
            output_file.create_dataset('data', data=data)

def generate_heatmap_frames(data):
    timesteps, height, width = data.shape
    c1, c2 = height, width

    fig, ax = plt.subplots(figsize=(c2 / 100, c1 / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    colors = ['darkred', 'red', 'lightcoral', 'white', 'lightskyblue', 'blue', 'darkblue']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', colors, N=100)

    frames = []

    for frame in range(timesteps):
        ax.clear()
        ax.set_axis_off()
        fig.patch.set_visible(False)
        ax.set_frame_on(False)

        heatmap = ax.imshow(data[frame], cmap=custom_cmap, interpolation='nearest')
        fig.canvas.draw()

        frame_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # Convert RGBA to RGB
        frames.append(frame_array.copy())

    plt.close(fig)
    return np.array(frames)


def process_shearflow_data_tracer(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    tracer = raw_data['t0_fields']['tracer']
    samples_count, sample_shape = tracer.shape[0], tracer.shape[1:]
    for i in range(samples_count):
        cleaned_data = generate_heatmap_frames(tracer[i])
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)


def process_shearflow_data(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    velocity = raw_data['t1_fields']['velocity']
    pressure = raw_data['t0_fields']['pressure']
    tracer = raw_data['t0_fields']['tracer']
    samples_count, sample_shape = tracer.shape[0], tracer.shape[1:]
    print(samples_count, sample_shape)
    cleaned_data = np.zeros(sample_shape + (4,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = tracer[i]
        cleaned_data[..., 1] = pressure[i]
        cleaned_data[..., 2:4] = velocity[i]
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_active_matter_data(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    concentration = raw_data['t0_fields']['concentration']
    velocity = raw_data['t1_fields']['velocity']
    D = raw_data['t2_fields']['D']
    E = raw_data['t2_fields']['E']
    samples_count, sample_shape = concentration.shape[0], concentration.shape[1:]
    print(samples_count, sample_shape)
    cleaned_data = np.zeros(sample_shape + (11,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = concentration[i]
        cleaned_data[..., 1:3] = velocity[i]
        cleaned_data[..., 3:7] = D[i].reshape(sample_shape + (4,))
        cleaned_data[..., 7:11] = E[i].reshape(sample_shape + (4,))
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_rayleigh_benard(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    buoyancy = raw_data['t0_fields']['buoyancy']
    pressure = raw_data['t0_fields']['pressure']
    velocity = raw_data['t1_fields']['velocity']
    samples_count, sample_shape = buoyancy.shape[0], buoyancy.shape[1:]
    print(samples_count, sample_shape)
    cleaned_data = np.zeros(sample_shape + (4,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = buoyancy[i]
        cleaned_data[..., 1] = pressure[i]
        cleaned_data[..., 2:4] = velocity[i]
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_viscoelastic_instability(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    pressure = raw_data['t0_fields']['pressure']
    c_zz = raw_data['t0_fields']['c_zz']
    velocity = raw_data['t1_fields']['velocity']
    C = raw_data['t2_fields']['C']
    samples_count, sample_shape = pressure.shape[0], pressure.shape[1:]
    print(samples_count, sample_shape)
    cleaned_data = np.zeros(sample_shape + (8,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = pressure[i]
        cleaned_data[..., 1] = c_zz[i]
        cleaned_data[..., 2:4] = velocity[i]
        cleaned_data[..., 4:8] = C[i].reshape(sample_shape + (4,))
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)
    

def process_turbulent_radiative_layer_2D(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    density = raw_data['t0_fields']['density']
    pressure = raw_data['t0_fields']['pressure']
    velocity = raw_data['t1_fields']['velocity']
    samples_count, sample_shape = density.shape[0], density.shape[1:]
    print(samples_count, sample_shape)
    cleaned_data = np.zeros(sample_shape + (4,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = density[i]
        cleaned_data[..., 1] = pressure[i]
        cleaned_data[..., 2:4] = velocity[i]
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_gray_scott_reaction_diffusion(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    A = raw_data['t0_fields']['A']
    B = raw_data['t0_fields']['B']
    samples_count, sample_shape = A.shape[0], A.shape[1:]
    print(samples_count, sample_shape)
    cleaned_data = np.zeros(sample_shape + (2,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = A[i]
        cleaned_data[..., 1] = B[i]
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_euler_multi_quadrants_openBC(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    energy = raw_data['t0_fields']['energy']
    density = raw_data['t0_fields']['density']
    pressure = raw_data['t0_fields']['pressure']
    momentum = raw_data['t1_fields']['momentum']  # This will have x and y components
    
    samples_count, sample_shape = energy.shape[0], energy.shape[1:]
    print(samples_count, sample_shape)
    
    cleaned_data = np.zeros(sample_shape + (5,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = energy[i]
        cleaned_data[..., 1] = density[i]
        cleaned_data[..., 2] = pressure[i]
        cleaned_data[..., 3:5] = momentum[i]  # x and y components
        
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_helmholtz_staircase(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    mask = raw_data['t0_fields']['mask']
    pressure_re = raw_data['t0_fields']['pressure_re']
    pressure_im = raw_data['t0_fields']['pressure_im']
    
    samples_count, sample_shape = pressure_re.shape[0], pressure_re.shape[1:]
    print(samples_count, sample_shape)
    
    cleaned_data = np.zeros(sample_shape + (3,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = pressure_re[i]
        cleaned_data[..., 1] = pressure_im[i]
        cleaned_data[..., 2] = mask[i]
        
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def process_acoustic_scattering(raw_file_path, cleaned_path, chunked=True):
    raw_data = h5py.File(raw_file_path, 'r')
    density = raw_data['t0_fields']['density'] # time-constant
    pressure = raw_data['t0_fields']['pressure']
    speed_of_sound = raw_data['t0_fields']['speed_of_sound'] # time-constant
    velocity = raw_data['t1_fields']['velocity']  # This will have x and y components
    
    samples_count, sample_shape = pressure.shape[0], pressure.shape[1:]
    time_n_dim = pressure.shape[1]
    print(samples_count, sample_shape)
    
    cleaned_data = np.zeros(sample_shape + (5,), dtype=np.float32)
    for i in range(samples_count):
        cleaned_data[..., 0] = np.expand_dims(density[i], axis=0).repeat(time_n_dim, axis=0)
        cleaned_data[..., 1] = pressure[i]
        cleaned_data[..., 2] = np.expand_dims(speed_of_sound[i], axis=0).repeat(time_n_dim, axis=0)
        cleaned_data[..., 3:5] = velocity[i]  # x and y components
        
        if chunked:
            save_chunked_data(cleaned_data, raw_file_path, cleaned_path, i)
        else:
            save_full_data(cleaned_data, raw_file_path, cleaned_path, i)

def load_image(img_path):
    return np.array(Image.open(img_path)).astype(np.float32) / 255.0

def process_phsgen(raw_file_path, cleaned_path, chunked=True):
    if not "input" in str(Path(raw_file_path)):
        return
        # not work -> implement on top level above

    input_sample_img_path = Path(raw_file_path)
    output_dir = Path(cleaned_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_sample_directory = os.path.dirname(input_sample_img_path)
    sample_name = os.path.basename(input_sample_img_path)
    target_sample_name = sample_name.replace("input", "target")

    # input_img = load_image(os.path.join(root_sample_directory, "input", sample_name))
    # target_img = load_image(os.path.join(root_sample_directory, "target", sample_name))
    input_img = load_image(os.path.join(root_sample_directory, sample_name))
    target_img = load_image(os.path.join(root_sample_directory, target_sample_name))

    # If just 2 dimensions, then expand (H, W) → (H, W, 1)
    if input_img.ndim == 2:
        input_img = np.expand_dims(input_img, axis=-1)
    if target_img.ndim == 2:
        target_img = np.expand_dims(target_img, axis=-1)

    # Combine at channel axis → (H, W, C)
    cleaned_data = np.concatenate([input_img, target_img], axis=-1)

    # Extract index from name
    idx = int(re.findall(r"\d+", sample_name)[0])

    save_name = "_".join(".".join(sample_name.split(".")[:-1]).split("_")[1:])+".hdf5"
    if chunked:
        save_chunked_data(cleaned_data, Path(save_name), output_dir, sample_index=idx, gzip_compression=True)
    else:
        save_full_data(cleaned_data, Path(save_name), output_dir, sample_index=idx, gzip_compression=True)

processor_mapping = {
    'shear_flow': process_shearflow_data,
    'active_matter': process_active_matter_data,
    'rayleigh_benard': process_rayleigh_benard,
    'viscoelastic_instability': process_viscoelastic_instability,
    'turbulent_radiative_layer_2D': process_turbulent_radiative_layer_2D,
    'gray_scott_reaction_diffusion': process_gray_scott_reaction_diffusion,
    'euler_multi_quadrants_openBC': process_euler_multi_quadrants_openBC,
    'euler_multi_quadrants_periodicBC': process_euler_multi_quadrants_openBC,
    'helmholtz_staircase': process_helmholtz_staircase,
    'acoustic_scattering': process_acoustic_scattering,
    'acoustic_scattering_discontinuous': process_acoustic_scattering,
    'acoustic_scattering_inclusions': process_acoustic_scattering,
    'acoustic_scattering_maze': process_acoustic_scattering,
    'physgen': process_phsgen,
}


