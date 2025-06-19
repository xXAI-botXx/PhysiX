# Cosmos Tokenizer: NeMo Framework Finetuning User Guide

Post-train the Cosmos Tokenizer using the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) to more accurately model previously unseen scenarios in your customer data, particularly for self-driving applications. By adapting the Cosmos Tokenizer to the specific characteristics and complexities of your in-house video content, you equip it to handle unique visual and temporal patterns that may have been missed during its initial pre-training. This enhanced modeling capability is essential for downstream diffusion models, which rely on the Tokenizer’s output to generate realistic physical scenes—ultimately boosting the performance and safety of your self-driving car systems.

## Model Support Matrix

The NeMo Framework currently supports the following Cosmos Tokenizer models. Review the available models for post-training.

| Model Name              | Model Ckpt           |
|-------------------------|----------------------------|
| Cosmos-1.0-Tokenizer-CV8x8x8          |  [HF Download](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-CV8x8x8)       |
| Cosmos-1.0-Tokenizer-DV8x16x16        |  [HF Download](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16)     |

For optimal performance, we recommend utilizing GPUs such as the H100-80GB or A100-80GB.
Note: Have a use case that would benefit from an alternative tokenizer? We'd love to hear from you. You can submit a request via a GitHub issue.

## Post-Training Support Matrix

Cosmos Tokenizer can be post-trained for a variety of Physical AI tasks. Review the following table for a list of available Physical AI post-training tasks:

| Post-training Task      | Support Status     |
|-------------------------|--------------------|
| General post-training and validation   | **Supported**      |

## Prerequisites

### 1. Review General Requirements

- System Configuration
  - **NVIDIA GPU and driver**: Ensure you have access to the 80G H100 or A100 to run the model(s).
  - **Containerization Platform**: We recommend using NVIDIA [NeMo Docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) Runtime (alternatively, you may use NVIDIA enroot).
- Get your [Hugging Face User Access Token](https://huggingface.co/docs/hub/en/security-tokens), which is required to obtain the Cosmos models for training and inference.
- Get your [Weights and Biases API Key](https://docs.wandb.ai/support/find_api_key/) for logging and tracking.

### 2. Clone the Cosmos Repository

```bash
git clone git@github.com:NVIDIA/Cosmos.git
```

### 3. Start the Container

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) supports post-training and inference for Cosmos Tokenizer models.

Run the following command to download and start the container:
   ```bash
   docker run --ipc=host -it --gpus=all \
    -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
    nvcr.io/nvidia/nemo:24.12.01 bash
   ```

### 4. Download Checkpoints

Follow the links provided in the Model Support Matrix to download the Cosmos Tokenizer checkpoints from Hugging Face. Detailed instructions for the download process are available on the Hugging Face page.


## Post-train

Post-training a Cosmos Tokenizer enables you to train the model to compress videos that are more specific to your Physical AI use case.

There are 3 steps to post-trainig: preparing a dataset, preprocessing the data, and post-training the model.

### 1. Prepare a Dataset

The first step is to prepare your dataset. Organize your data into a folder containing multiple video tars, each contains MP4 format videos (preferably at least 720p resolution). The recommended folder structure is as follows:

- `000000.tar`
  - `1.mp4`
  - `2.mp4`
- `000001.tar`
  - `3.mp4`
  - `4.mp4`

Here, 000000.tar and 000001.tar represent separate shards, and you may include additional shards as needed.

### 3. Post-train the Model

The third step is to post-train the Cosmos tokenizer using the NeMo Framework.

#### Run the Post-training Script

Complete the following steps to post-train the Cosmos tokenizer Cosmos-1.0-Tokenizer-CV8x8x8.

1. Install the dependencies under cosmos1/models/tokenizer/nemo:
   ```bash
    pip install megatron-energon==4.0.0 pyav
    pip install git+https://github.com/NVIDIA/NeMo-Run.git
    pip install moviepy==1.0.3 imageio
   ```

2. Run the following command to post-train Cosmos-1.0-Tokenizer-CV8x8x8:
   ```bash
    export CKPT_PTH="<path/to/your/HF/checkpoints/folder>"
    export DATA="<path/to/your/data>"

    # Optionally, you can monitor training progress with Weights and Biases (wandb).
    export WANDB_API_KEY="</your/wandb/api/key>"
    export WANDB_PROJECT_NAME="cosmos-diffusion-nemo-post-training"
    export WANDB_RUN_ID="cosmos_diffusion_7b_text2world"

   torchrun --nproc-per-node 8  cosmos1/models/tokenizer/nemo/train_tokenizer.py --yes \
    data.path=$DATA \
    model.jit_ckpt_pth=$CKPT_PTH \
    model.model="Cosmos-1.0-Tokenizer-CV8x8x8"
   ```

##### Configurable Hyperparameters

For a comprehensive list of configurable hyperparameters, please refer to the `train_tokenizer.py` script. The script supports four major configuration components:

1. **model**: Select a model for post-training and pass the model checkpoint.
2. **data**: Define batch size and dataloader related hyper-parameters.
3. **trainer**: Define the training loop.
4. **optim**: Specify the post-training optimizer hyperparameters.

You can configure any hyperparameter of these four components by setting the value in the launch script using the following format:

```bash
model.jit_ckpt_pth=<your/desired/path>
trainer.max_epochs=<your/desired/epochs>
```

Adjust the values as needed to suit your training requirements. After a few hundred iterations, you should observe that the 'loss' reported in Weights & Biases (`wandb`) starts decreasing.


<p align="center">
  <img src="./assets/loss.png" alt="Image description" width="50%">
</p>
