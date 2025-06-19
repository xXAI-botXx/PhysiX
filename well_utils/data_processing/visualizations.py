from argparse import ArgumentParser
import numpy as np
import h5py


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from well_utils.data_processing.helpers import save_numpy_as_mp4

def hd5f_to_mp4(args):
    with h5py.File(args.file_path, 'r') as hdf_file:
        data = np.array(hdf_file['data'])
    save_numpy_as_mp4(data, args.save_path, fps=25)


def create_video_heatmap(data, save_path=None, fps=30, channel_names=None):
    def _validate_and_prepare_data(data, channel_names):
        if len(data.shape) not in (4, 5):
            raise ValueError("Input data must have shape (channels, timesteps, height, width) or (batch, channels, timesteps, height, width)")
        
        if len(data.shape) == 4:
            data = np.expand_dims(data, axis=0)
        
        batch_size, channels, timesteps, height, width = data.shape
        
        if channel_names and len(channel_names) != channels:
            raise ValueError(f"Length of channel_names ({len(channel_names)}) must match the number of channels ({channels})")
        
        return data, batch_size, channels, timesteps, height, width

    def _setup_figure(batch_size, channels, height, width, channel_names):
        base_height = (height / 100) * batch_size
        fig_width = (width * channels) / 100
        
        fig_height = base_height + (0.3 if channel_names else 0)
        
        fig, axes = plt.subplots(
            batch_size, channels,
            figsize=(fig_width, fig_height),
            dpi=100,
            squeeze=False,
            gridspec_kw={'hspace': 0}
        )
        fig.patch.set_facecolor('black')
        return fig, axes, base_height

    def _configure_figure(fig, axes, batch_size, channels, channel_names, base_height):
        if channel_names:
            total_height = fig.get_size_inches()[1]
            top = (base_height / total_height) * 0.97
        else:
            top = 1.0

        fig.subplots_adjust(left=0, right=1, top=top, bottom=0, wspace=0.02)
        
        if batch_size > 1:
            for i in range(batch_size - 1):
                y_pos = axes[i, 0].get_position().y0
                line = Line2D([0, 1], [y_pos, y_pos], 
                            color='white', 
                            linewidth=1.5,
                            transform=fig.transFigure)
                fig.add_artist(line)

    def _create_colormap():
        colors = ['darkred', 'red', 'lightcoral', 'white', 'lightskyblue', 'blue', 'darkblue']
        return mcolors.LinearSegmentedColormap.from_list('red_blue', colors, N=100)

    # Main execution
    data, batch_size, channels, timesteps, height, width = _validate_and_prepare_data(data, channel_names)
    
    # Compute min and max for each channel across all batches and timesteps
    channel_vmin = []
    channel_vmax = []
    for j in range(channels):
        channel_data = data[:, j, ...]  # Shape: (batch_size, timesteps, height, width)
        channel_vmin.append(np.min(channel_data))
        channel_vmax.append(np.max(channel_data))
    
    fig, axes, base_height = _setup_figure(batch_size, channels, height, width, channel_names)
    _configure_figure(fig, axes, batch_size, channels, channel_names, base_height)
    custom_cmap = _create_colormap()

    # Optimization: Create image objects once and update data later
    images = [[None for _ in range(channels)] for _ in range(batch_size)]
    
    # Initialize the plot with first frame
    for i in range(batch_size):
        for j in range(channels):
            ax = axes[i, j]
            ax.set_axis_off()
            ax.set_frame_on(False)
            if i == 0 and channel_names is not None:
                ax.set_title(channel_names[j], color='white', fontsize=24, pad=2)
            images[i][j] = ax.imshow(data[i, j, 0], cmap=custom_cmap, 
                      vmin=channel_vmin[j], vmax=channel_vmax[j],
                      interpolation='nearest', aspect='auto')

    # Optimized update function that only changes the data, not recreating plots
    def _update(frame):
        for i in range(batch_size):
            for j in range(channels):
                images[i][j].set_array(data[i, j, frame])
        return [img for row in images for img in row]

    if save_path:
        # Use more efficient animation settings
        plt.rcParams['animation.embed_limit'] = 2**128  # Avoid any warnings/limits
        ani = animation.FuncAnimation(fig, _update, frames=timesteps, blit=True)
        
        # Configure writer with higher bitrate and efficient codec
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000, codec='h264_nvenc' if _has_nvenc() else 'libx264')
        ani.save(save_path, writer=writer)
        plt.close(fig)
        return None
    else:
        # Optimized video array generation
        frames = np.empty((timesteps, *fig.canvas.get_width_height()[::-1], 3), dtype=np.uint8)
        for frame in range(timesteps):
            _update(frame)
            fig.canvas.draw()
            # More efficient buffer extraction
            buffer = fig.canvas.buffer_rgba()
            img = np.asarray(buffer)[..., :3]
            frames[frame] = img
        plt.close(fig)
        return frames

def _has_nvenc():
    """Check if NVENC is available for hardware acceleration"""
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return 'h264_nvenc' in result.stdout
    except:
        return False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_path", required=True, type=str, help="The path to the HDF5 file")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the MP4 file")
    args = parser.parse_args()

    data = np.array(h5py.File(args.file_path, 'r')['data'])
    data = np.transpose(data, (3, 0, 1, 2))
    print(data.shape)
    print("Max:", data.max(), "Mean:", data.mean(), "Min:", data.min())
    create_video_heatmap(data, save_path=args.save_path, fps=25)

    # hd5f_to_mp4(args)