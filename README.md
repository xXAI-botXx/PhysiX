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

## Installation

<!--
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
-->
1. Setup env
  ```bash
  git clone https://github.com/your-org/PhysiX.git
  cd PhysiX

  conda env create -f environment.yaml
  conda activate physix

  pip install -e .
  ```
2. Docker setup
  ```bash
  # --- Docker ---
  # Make Sure Docker is installed
  docker --version
  which docker

  # If not run:
  sudo apt update
  sudo apt install -y \
      ca-certificates \
      curl \
      gnupg \
      lsb-release

  # Add Docker’s official GPG key:
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

  # Set up the Docker repository:
  echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  # Install Docker Engine:
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  # --- Nvidia Container Toolkit ---
  # Make sure nvidia container toolkit is installed
  dpkg -l | grep nvidia-container-toolkit

  # Else install it -> see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

  # --- Cosmos Container ---
  cd ~/src/PhysiX
  docker build --no-cache -t physix -f Dockerfile .
  ```
3. Download data
  ```bash
  conda activate physix
  python physgen_dataset.py --output_path ./datasets/physgen/raw/train --variation sound_reflection --input_type osm --output_type standard --data_mode train

  python physgen_dataset.py --output_path ./datasets/physgen/raw/test --variation sound_reflection --input_type osm --output_type standard --data_mode test

  python physgen_dataset.py --output_path ./datasets/physgen/raw/val --variation sound_reflection --input_type osm --output_type standard --data_mode validation
  ```
4. Preprocess Data (convert to hf5 and input and target together)
  ```bash
  python -m well_utils.data_processing.process_dataset \
    physgen \
    --raw_data_path    ./datasets/physgen/raw/train \
    --cleaned_data_path ./datasets/physgen/cleaned/train
  ```
  ```bash
  python -m well_utils.data_processing.process_dataset \
    physgen \
    --raw_data_path    ./datasets/physgen/raw/test \
    --cleaned_data_path ./datasets/physgen/cleaned/test
  ```
  ```bash
  python -m well_utils.data_processing.process_dataset \
    physgen \
    --raw_data_path    ./datasets/physgen/raw/val \
    --cleaned_data_path ./datasets/physgen/cleaned/val
  ```
5. Normalization Stats
  ```bash
  python -m well_utils.data_processing.normalization.calculate_stats \
  --input_dir ./datasets/physgen/cleaned/train \
  --output_path ./datasets/physgen/cleaned/train/normalization_stats.json
  ```
  ```bash
  python -m well_utils.data_processing.normalization.calculate_stats \
  --input_dir ./datasets/physgen/cleaned/test \
  --output_path ./datasets/physgen/cleaned/test/normalization_stats.json
  ```
  ```bash
  python -m well_utils.data_processing.normalization.calculate_stats \
  --input_dir ./datasets/physgen/cleaned/val \
  --output_path ./datasets/physgen/cleaned/val/normalization_stats.json
  ```
6. Normalization
  ```bash
  python -m well_utils.data_processing.normalization.normalize \
  --input_dir  ./datasets/physgen/cleaned/train \
  --output_dir ./datasets/physgen/normalized/train \
  --stats_path ./datasets/physgen/cleaned/train/normalization_stats.json \
  --normalization_type standard
  ```
  ```bash
  python -m well_utils.data_processing.normalization.normalize \
  --input_dir  ./datasets/physgen/cleaned/test \
  --output_dir ./datasets/physgen/normalized/test \
  --stats_path ./datasets/physgen/cleaned/test/normalization_stats.json \
  --normalization_type standard
  ```
  ```bash
  python -m well_utils.data_processing.normalization.normalize \
  --input_dir  ./datasets/physgen/cleaned/val \
  --output_dir ./datasets/physgen/normalized/val \
  --stats_path ./datasets/physgen/cleaned/val/normalization_stats.json \
  --normalization_type standard
  ```

<!--
7. Train Tokenizer/Embedding -> Continuous VAE, because we have continuous data and not discrete
  1. Make an Wandb account, create an API Token and paste the token into the command `-e WANDB_API_KEY="YOUR_API_TOKEN"` -> keep your token always hidden/secret! Link: https://wandb.ai/
  2. Make a hugging-face account (https://huggingface.co/) and go to https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-CV8x8x8/ -> you might want to accept/agree the license (it's easy to overlook, so look carefully). Create also an API Token and copy it, then login with your token using (use 'Add token as git credential? (Y/n)' y):
    ```bash
    git config --global credential.helper store
    hf auth login
    ```
  3. Download Tokenizer Checkpoint (from huggingface)
    Make folders before:
    ```bash
    mkdir -p /ssd0/tippolit/physix/checkpoints
    mkdir -p /ssd0/tippolit/physix/checkpoints/physgen/autoencoder
    mkdir -p /ssd0/tippolit/physix/checkpoints/physgen/continuous-vae
    mkdir -p /home/tippolit/src/PhysiX/physix_checkpoints/continuous-vae
    ```
    ```bash
    cd /ssd0/tippolit/physix/checkpoints/physgen
    # or
    cd ~/src/PhysiX && mkdir physix_checkpoints
    git clone https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-CV8x8x8
    ```
  4. Start Tokenizer Training:
    Get the right container permissions for user tippolit (or whatever your username is)
    ```bash
    chmod -R a+rwX /ssd0/tippolit/physix
    ```
    ```bash
    docker run --gpus '"device=0"' --runtime=nvidia -d \
    --shm-size=8g \
    --rm \
    -v /home/tippolit/src/PhysiX:/workspace \
    -v /ssd0/tippolit/physix/checkpoints:/checkpoints \
    -e WANDB_API_KEY="5171efce2df498fa22f2f80de3263a5f36dbb7ec" \
    --name physix-tokenizer-train-run \
    physix \
    bash -c "( \
        echo '--- 1. Listing /workspace ---' && ls -l /workspace ; \
        echo -e '\n--- 2. Listing /checkpoints ---\n' && ls -l /checkpoints ; \
        echo -e '\n--- 3. Listing /checkpoints/physgen ---\n' && ls -l /checkpoints/physgen ; \
        echo -e '\n--- 4. Listing Checkpoint Dir ---\n' && ls -l /checkpoints/physgen/Cosmos-1.0-Tokenizer-CV8x8x8 ; \
      ) > docker_info.log 2>&1 && \
      nohup torchrun --nproc_per_node=1 --master_port=12341 \
      -m cosmos1.models.tokenizer.training.general \
      --train_data_path ./datasets/physgen/normalized/train \
      --val_data_path ./datasets/physgen/normalized/val \
      --autoencoder_path /checkpoints/physgen/Cosmos-1.0-Tokenizer-CV8x8x8/mean_std.pt \
      --checkpoint_dir /checkpoints/physgen/continuous-vae \
      --batch_size 4 \
      --epochs 5000 \
      --save_every_n_epochs 5 \
      --visual_log_interval 5 \
      --data_resolution    256 256 \
      --grad_accumulation_steps 2 \
      --clip_grad_norm     2.0 \
      --stats_path         ./datasets/physgen/cleaned/train/normalization_stats.json \
      --beta               0.01 \
      > train_tokenizer.log 2>&1"
    ``` 
    Or without external space:
    ```bash
    docker run --gpus '"device=0"' --runtime=nvidia -d \
    --shm-size=8g \
    --rm \
    -v /home/tippolit/src/PhysiX:/workspace \
    -e WANDB_API_KEY="5171efce2df498fa22f2f80de3263a5f36dbb7ec" \
    --name physix-tokenizer-train-run \
    physix \
    bash -c "nohup torchrun --nproc_per_node=1 --master_port=12341 \
      -m cosmos1.models.tokenizer.training.general \
      --train_data_path ./datasets/physgen/normalized/train \
      --val_data_path ./datasets/physgen/normalized/val \
      --autoencoder_path ./physix_checkpoints/Cosmos-1.0-Tokenizer-CV8x8x8/encoder.jit \
      --checkpoint_dir ./physix_checkpoints/continuous-vae \
      --batch_size 4 \
      --epochs 5000 \
      --save_every_n_epochs 5 \
      --visual_log_interval 5 \
      --data_resolution    256 256 \
      --grad_accumulation_steps 2 \
      --clip_grad_norm     2.0 \
      --stats_path         ./datasets/physgen/cleaned/train/normalization_stats.json \
      --beta               0.01 \
      > train_tokenizer.log 2>&1"
    ``` 
    Stopping:
    ```bash
    docker stop physix-tokenizer-train-run && docker rm -f /physix-tokenizer-train-run
    ```
    Check Logs:
    ```bash
    docker logs -f physix-tokenizer-train-run
    ```
8. Apply Tokenizer
  ```bash
  python -m cosmos1.models.tokenizer.lobotomize.inflate_channels_continuous \
  --weights /ssd0/tippolit/physix/checkpoints/physgen/continuous-vae/Cosmos-1.0-Tokenizer-CV8x8x8/autoencoder.jit \
  --original_channels 3 \
  --new_channels 1 \
  --frames 2 \
  --height 256 \
  --width 256
  ```
-->
*Are there more steps needed?*
  


## Training

FIXME -> adjust project_config.yaml
      -> Use normalization path as data_path, try out

```bash
torchrun --master_port 12345 --nproc-per-node 1 -m cosmos1.models.autoregressive.nemo.post_training.general \
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

## Inference

...

<br><br>

---
### Original Content

---

<br><br>

# PhysiX: A Foundation Model for Physics Simulations

## Abstract

Foundation models have achieved remarkable success across video, image, and language domains. By scaling up the number of parameters and training datasets, these models acquire generalizable world knowledge and often surpass task-specific approaches. However, such progress has yet to extend to the domain of physics simulation. A primary bottleneck is data scarcity: while millions of images, videos, and textual resources are readily available on the internet, the largest physics simulation datasets contain only tens of thousands of samples. This data limitation hinders the use of large models, as overfitting becomes a major concern. As a result, physics applications typically rely on small models, which struggle with long-range prediction due to limited context understanding. We introduce PhysiX, the first large-scale foundation model for physics simulation. PhysiX is a 4.5B parameter autoregressive generative model. We show that PhysiX effectively addresses the data bottleneck, outperforming task-specific baselines under comparable settings as well as the previous absolute state-of-the-art approaches on The Well benchmark.




## 2. Installation

```bash
git clone https://github.com/your-org/PhysiX.git
cd PhysiX

conda env create -f environment.yaml
conda activate physix

pip install -e .
```

### 2.1 Configuration

After cloning the repository and installing dependencies, configure your project paths by editing `project_config.yaml`:

```yaml
raw_data_path: /path/to/raw/data
cleaned_data_path: /path/to/cleaned/data
normalized_data_path: /path/to/normalized/data
checkpoint_dir: /path/to/checkpoints
embeddings_dir: /path/to/embeddings
results_dir: /path/to/results
tokenizer_path: /path/to/tokenizer
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

Optionally inflate/deflate the input and output channels of the Cosmos AE to preserve pretrained weights to finetune on simulation data

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

## 4. Citation

```bibtex
@article{nguyen2025physix,
  title={PhysiX: A Foundation Model for Physics Simulations},
  author={Nguyen, Tung and Koneru, Arsh and Li, Shufan and others},
  journal={arXiv preprint arXiv:2506.17774},
  year={2025}
}
```

## 5. Acknowledgments

This project is adapted from [Cosmos](https://github.com/nvidia-cosmos/cosmos-predict1), an open-source framework developed by NVIDIA