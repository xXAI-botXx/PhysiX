Current Issues:
- Env installation under Windows -> transformer-engine
- PhysiX does not provide official Checkpoints (on date: 15.07.2025)

# PhysiX on Physgen Benchmark

PhysiX:
- [Paper](https://arxiv.org/abs/2506.17774)
- [Original Code](https://github.com/ArshKA/PhysiX)

Physgen (benchmark):
- [Paper](https://arxiv.org/abs/2503.05333)
- [Dataset](https://huggingface.co/papers/2503.05333)

## 1. Installation

(Changed from the original)
```
# Clone repository
git clone https://github.com/xXAI-botXx/PhysiX.git
cd PhysiX

# Create & activate Conda environment
conda env create -f ./environment.yaml
conda activate physix
conda install -c conda-forge cmake -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers tensorboard sympy timm tqdm scikit-learn pyyaml pydantic datasets pillow wandb ipython ipykernel scikit-image pytorch-msssim pandas prime_printer shapely ipykernel tqdm kornia numba iopath nemo_run transformer_engine>=1.4
pip install -e .
```
<br><br>

**Configuration**

After cloning the repository and installing dependencies, configure your project paths by editing `project_config.yaml` (replace placeholders accordingly):

```yaml
raw_data_path: /path/to/raw/data
cleaned_data_path: /path/to/cleaned/data
normalized_data_path: /path/to/normalized/data
checkpoint_dir: /path/to/checkpoints
embeddings_dir: /path/to/embeddings
results_dir: /path/to/results
tokenizer_path: /path/to/tokenizer
cache_dir: /path/to/cache
```


## 3. Usage Overview

### Data Processing

Normalizes data from The Well and standardizes channel formats.

#### 3.1 Data Processing

```bash
python -m well_utils.data_processing.process_dataset \
  <dataset_name> \
  --raw_data_path    /data/raw/datasets/ \
  --cleaned_data_path /data/cleaned/<dataset_name>/
```

#### 3.2 Data Normalization

```bash
# Compute stats
python -m well_utils.data_processing.normalization.calculate_stats \
  --input_dir  /data/cleaned/<dataset>/ \
  --output_path /data/normalized/<dataset>/normalization_stats.json

# Normalize (standard or minmax)
python -m well_utils.data_processing.normalization.normalize \
  --input_dir  /data/cleaned/<dataset>/ \
  --output_dir  /data/normalized/<dataset>/ \
  --stats_path  /data/normalized/<dataset>/normalization_stats.json \
  --normalization_type standard --delete
```

#### 3.3 Tokenizer Inflation & Training

Optionally inflate/deflate the input and output channels of the Cosmos AE to preserve pretrained weights. Finetune on simulation data

**Discrete Channels**:
```bash
python -m cosmos1.models.autoregressive.tokenizer.lobotomize.inflate_channels_discrete \
  --autoencoder_path  /checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16 \
  --original_channels 3 --new_channels 11 \
  --dimensions 33 256 256
```

**Continuous Channels**:
```bash
python -m cosmos1.models.tokenizer.lobotomize.inflate_channels_continuous \
  --weights            /checkpoints/Cosmos-1.0-Tokenizer-CV8x8x8/autoencoder.jit \
  --original_channels 3 --new_channels 4 \
  --frames 33 --height 256 --width 256
```

##### Specialized Tokenizer Training

**Continuous VAE**:
```bash
torchrun --nproc_per_node 8 -m cosmos1.models.tokenizer.training.general \
  --train_data_path    /data/normalized/<DATASET>/train \
  --val_data_path      /data/normalized/<DATASET>/valid \
  --autoencoder_path   /checkpoints/Cosmos-1.0-Tokenizer-CV8x8x8/vae_<new_channels>c.pt \
  --checkpoint_dir     /checkpoints/tokenizers/<DATASET>/continuous \
  --batch_size         4 \
  --epochs             5000 \
  --save_every_n_epochs 5 \
  --visual_log_interval 5 \
  --data_resolution    256 256 \
  --grad_accumulation_steps 2 \
  --clip_grad_norm     2.0 \
  --stats_path         /data/normalized/<DATASET>/normalization_stats.json \
  --beta               0.01
```

**Discrete VQ-VAE**:
```bash
python -m cosmos1.models.autoregressive.tokenizer.lobotomize.inflate_channels_discrete \
  --autoencoder_path     /checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16 \
  --original_channels    3 \
  --new_input_channels   <new_channels> \
  --new_output_channels  <new_channels> \
  --dimensions           33 256 256

python cosmo_lightning/train_universal_vae.py \
  --config lightning_configs/pretrained_discrete<DATASET>.yaml
```

##### Universal Tokenizer Training

```bash
python cosmo_lightning/train_universal_vae_distributed.py \
  --config lightning_configs/universal_vae_dvd_padded_distributed.yaml
```

#### 3.4 Autoregressive Model Fine-tuning

```bash
torchrun --master_port 12345 --nproc-per-node 8 -m cosmos1.models.autoregressive.nemo.post_training.general \
  --data_path            /data/embeddings/<dataset>/ \
  --model_path           nvidia/Cosmos-1.0-Autoregressive-4B \
  --index_mapping_dir    /checkpoints/indices/PROJECT \
  --split_string         90,5,5 \
  --log_dir              /checkpoints/logs/PROJECT \
  --max_steps            8000 \
  --save_every_n_steps   1000 \
  --tensor_model_parallel_size 8 \
  --global_batch_size    8 \
  --micro_batch_size     1 \
  --latent_shape         4 64 64 \
  --lr                   1e-4
```

#### 3.5 Inference & Evaluation

```bash
PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/evaluation/general.py \
  --batch_input_path    /data/normalized/<DATASET>/test/ \
  --checkpoint_dir      /checkpoints/finetuned/ \
  --ar_model_dir        Cosmos-1.0-Autoregressive-4B \
  --tokenizer_path      /checkpoints/tokenizers/<DATASET>/last.pth \
  --channel_stats_path  /data/normalized/<DATASET>/normalization_stats.json \
  --dimensions          256 256 \
  --context_len         9 \
  --random_eval_samples 10 \
  --visualize_interval  1 \
  --output_dir          results/<DATASET>/ \
  --compression_ratio   4 8 8
```

## 4. CLI Reference

- **Data Processing**: `well_utils.data_processing.process_dataset`  
- **Normalization**: `well_utils.data_processing.normalization.normalize`  
- **Tokenizer Inflation**: `cosmos1.models.autoregressive.tokenizer.lobotomize.inflate_channels_*`  
- **Tokenizer Training**: `cosmo_lightning/train_universal_vae_distributed.py`  
- **AR Training**: `cosmos1.models.autoregressive.nemo.post_training.general`  
- **Evaluation**: `cosmos1.models.autoregressive.evaluation.general`  

Run any module with `-h` to view detailed flags.

## 5. Results & Visualizations

- **Checkpoints**: saved to `<log_dir>`  
- **Metrics**: `metrics.json`, `all_metrics.json` in `<output_dir>`  
- **Generated Arrays**: `.npz` files in `<output_dir>/arrays/`  
- **Visualizations**: MP4 heatmaps in `<output_dir>`

## 6. Citation

```bibtex
@misc{nguyen2025uniphy,
  title={PhysiX: A Foundation Model for Physics Simulations},
  author={Tung Nguyen and Arsh Koneru and Shufan Li and Aditya Grover},
  year={2025},
}
```

## 7. Acknowledgments

This project is adapted from [Cosmos](https://github.com/nvidia-cosmos/cosmos-predict1), an open-source framework developed by NVIDIA
