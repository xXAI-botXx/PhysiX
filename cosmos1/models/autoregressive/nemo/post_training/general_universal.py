# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import random
from datetime import datetime
from argparse import ArgumentParser
from glob import glob

import torch
from huggingface_hub import snapshot_download
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections.llm.api import _use_tokenizer
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, PreemptionCallback
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar

# from cosmos1.models.autoregressive.nemo.cosmos import CosmosConfig4B, CosmosConfig12B, CosmosModel
from cosmo_lightning.data.sequence_datamodule import SequenceDatamodule
from cosmo_lightning.data.sequence_multi_datamodule import SequenceMultiDatamodule
from cosmos1.models.autoregressive.nemo.custom_cosmos import CustomCosmosModel, CustomCosmosConfig4B, CustomCosmosConfig12B, CustomCosmosConfig2B, CustomCosmosConfig600M
from cosmos1.models.autoregressive.nemo.custom_strategy import CustomMegatronStrategy
from cosmos1.models.autoregressive.nemo.custom_lr_scheduler import CustomCosineAnnealingScheduler
from config import EMBEDDINGS_DIR


DATA_METADATA = {
    "active_matter": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/active_matter_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/active_matter_AR_discrete",
        "latent_shapes": [4, 32, 32],
        "channel_names": ["concentration_AM", "velocity_x_AM", "velocity_y_AM", "D_xx_AM", "D_xy_AM", "D_yx_AM", "D_yy_AM", "E_xx_AM", "E_xy_AM", "E_yx_AM", "E_yy_AM"]
    },
    "shear_flow": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/shear_flow_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/shear_flow_AR_discrete",
        "latent_shapes": [4, 32, 64],
        "channel_names": ["tracer_SF", "pressure_SF", "velocity_x_SF", "velocity_y_SF"]
    },
    "rayleigh_benard": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/rayleigh_benard_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/rayleigh_benard_AR_discrete",
        "latent_shapes": [4, 64, 16],
        "channel_names": ["buoyancy_RB", "pressure_RB", "velocity_x_RB", "velocity_y_RB"]
    },
    "acoustic_scattering_maze": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/acoustic_scattering_maze_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/acoustic_scattering_maze_AR_discrete",
        "latent_shapes": [4, 32, 32],
        "channel_names": ["density_ASM", "pressure_ASM", "speed_of_sound_ASM", "velocity_x_ASM", "velocity_y_ASM"]
    },
    "gray_scott_reaction_diffusion": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/gray_scott_reaction_diffusion_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/gray_scott_reaction_diffusion_AR_discrete",
        "latent_shapes": [4, 16, 16],
        "channel_names": ["A_GS", "B_GS"]
    },
    "helmholtz_staircase": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/helmholtz_staircase_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/helmholtz_staircase_AR_discrete",
        "latent_shapes": [4, 128, 32],
        "channel_names": ["pressure_re_HS", "pressure_im_HS", "mask_HS"]
    },
    "turbulent_radiative_layer_2D": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/turbulent_radiative_layer_2D_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/turbulent_radiative_layer_2D_AR_discrete",
        "latent_shapes": [4, 16, 48],
        "channel_names": ["density_TR", "pressure_TR", "velocity_x_TR", "velocity_y_TR"]
    },
    "viscoelastic_instability": {
        "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_one_device/viscoelastic_instability_AR_discrete",
        # "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/embeddings_stage_2_one_device/viscoelastic_instability_AR_discrete",
        "latent_shapes": [4, 64, 64],
        "channel_names": ["pressure_VI", "c_zz_VI", "velocity_x_VI", "velocity_y_VI", "C_xx_VI", "C_xy_VI", "C_yx_VI", "C_yy_VI"]
    }
}


def extract_latest_run_id(root_dir, logger_name):
    """
    Extract the ID of the latest wandb run based on the directory names.
    
    Args:
        root_dir (str): The root directory path
        logger_name (str): The name of the logger
        
    Returns:
        str: The ID of the latest run, or None if no runs found
    """
    wandb_dir = os.path.join(root_dir, logger_name, 'wandb')
    
    if not os.path.exists(wandb_dir):
        return None
    
    run_dirs = [
        d for d in os.listdir(wandb_dir) 
        if os.path.isdir(os.path.join(wandb_dir, d)) and d.startswith('run-')
    ]
    
    if not run_dirs:
        return None
    
    # Parse the directory names to extract date, time, and ID
    run_info = []
    pattern = r'run-(\d{8})_(\d{6})-(\w+)'
    
    for dir_name in run_dirs:
        match = re.match(pattern, dir_name)
        if match:
            date_str, time_str, run_id = match.groups()
            # Convert date and time strings to datetime object
            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            run_info.append((timestamp, run_id, dir_name))
    
    if not run_info:
        return None
    
    # Sort by timestamp (latest first)
    run_info.sort(reverse=True)
    
    # Return the ID of the latest run
    return run_info[0][1]


def main(args):
    torch.set_float32_matmul_precision('medium')


    if args.model_size == "4B":
        config = CustomCosmosConfig4B()
    elif args.model_size == "12B":
        config = CustomCosmosConfig12B()
    elif args.model_size == "2B":
        config = CustomCosmosConfig2B()
    elif args.model_size == "600M":
        config = CustomCosmosConfig600M()
    else:
        raise NotImplementedError
    
    
    load_optim_state = False
    if os.path.exists(os.path.join(args.root_dir, args.exp_name, "checkpoints")):
        # get all dirs whose name includes "last"
        dirs = glob(os.path.join(args.root_dir, args.exp_name, "checkpoints", "*last*"))
        
        if dirs:
            # Function to extract epoch and step from directory name
            def extract_epoch_step(dir_path):
                basename = os.path.basename(dir_path)
                epoch_match = re.search(r'epoch=(\d+)', basename)
                step_match = re.search(r'step=(\d+)', basename)
                
                if epoch_match and step_match:
                    epoch = int(epoch_match.group(1))
                    step = int(step_match.group(1))
                    return epoch, step
                return -1, -1  # Return defaults if pattern doesn't match
            
            # Sort directories by epoch (primary) and step (secondary)
            dirs.sort(key=extract_epoch_step, reverse=True)
            
            # Select the first one (highest epoch and step)
            args.model_path = dirs[0]
            print(f"Found checkpoint: {args.model_path}")
            load_optim_state = True
            state_dict = torch.load(os.path.join(args.model_path, "weights", "common.pt"))
        else:
            state_dict = None
    else:
        state_dict = None

    if args.model_path in ["nvidia/Cosmos-1.0-Autoregressive-4B", "nvidia/Cosmos-1.0-Autoregressive-12B"]:
        args.model_path = os.path.join(snapshot_download(args.model_path, allow_patterns=["nemo/*"]), "nemo")

    config.latent_shape = args.latent_shape
    model = CustomCosmosModel(config)
    
    data_module = SequenceMultiDatamodule(
        # dataset_name="active_matter",
        # root_dir=DATA_METADATA["active_matter"]["root_dir"],
        # latent_shapes=args.latent_shape,
        data_metadata=DATA_METADATA,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        init_global_step=state_dict["global_step"] if state_dict else 0,
        init_consumed_samples=(state_dict["global_step"] * state_dict['datamodule_hyper_parameters']['global_batch_size']) if state_dict else 0,
        num_workers=16,
        pin_memory=False,
        seed=args.seed,
    )
    # data_module.setup()
    # len_train_ds = len(data_module.data_train)
    # n_steps_per_epoch = len_train_ds // args.global_batch_size
    # max_steps = n_steps_per_epoch * args.max_epochs

    # Finetune is the same as train (Except train gives the option to set tokenizer to None)
    # So we use it since in this case we dont store a tokenizer with the model
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.root_dir, args.exp_name, "checkpoints"),
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{step}-{val_loss:.2f}",
        # every_n_train_steps=args.val_check_interval,
        save_on_train_epoch_end=False,
        save_top_k=2,
        save_last=True,
    )
    model_checkpoint_callback.async_save = True
    
    lightning_trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=CustomMegatronStrategy(
            tensor_model_parallel_size=(args.tp),
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_strictness=False,
        ),
        val_check_interval=args.val_check_interval,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        num_sanity_val_steps=0,
        limit_val_batches=10,
        log_every_n_steps=1,
        callbacks=[
            model_checkpoint_callback,
            PreemptionCallback(),
            LearningRateMonitor(logging_interval='step'),
            TQDMProgressBar(),
        ],
    )
    
    if args.model_path.lower() in ["none", "null"]:
        args.model_path = None
    
    resume = None
    if args.model_path is not None:
        resume=nl.AutoResume(
            restore_config=RestoreConfig(path=args.model_path, load_optim_state=load_optim_state),
            resume_if_exists=True,
            resume_ignore_no_checkpoint=False,
            resume_past_end=True,
        )
        resume.setup(lightning_trainer, model)
    
    latest_run_id = extract_latest_run_id(args.root_dir, args.exp_name)
    if latest_run_id is not None:
        print("Resuming wandb run with ID:", latest_run_id)
        wdb = WandbLogger(
            name=args.exp_name,
            project="physics_sim",
            save_dir=os.path.join(args.root_dir, args.exp_name),
            id=latest_run_id,
            resume='must'
        )
    else:
        print("Starting a new wandb run")
        wdb = WandbLogger(
            name=args.exp_name,
            project="physics_sim",
            save_dir=os.path.join(args.root_dir, args.exp_name),
        )
    
    log = nl.NeMoLogger(
        wandb=wdb,
        log_dir=os.path.join(args.root_dir, args.exp_name, "logs"),
    )
    app_state = log.setup(
        lightning_trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False) if resume is not None else False,
        task_config=None,
    )
    
    optim=nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=args.lr,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
        ),
        lr_scheduler=CustomCosineAnnealingScheduler(
            max_steps=args.max_steps,
            warmup_steps=int(0.1 * args.max_steps),
            constant_steps=0,
            min_lr=1e-7,
            lr_scheduler_state_dict=state_dict["lr_schedulers"][0] if state_dict else None,
        ),
    )
    optim.connect(model)
    
    _use_tokenizer(model, data_module, 'data')
    
    lightning_trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, default=EMBEDDINGS_DIR, help="The path to the input videos"
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="The name of the experiment"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for random number generation"
    )
    parser.add_argument(
        "--model_size", default="4B", type=str, help="The size of the model"
    )
    parser.add_argument(
        "--model_path", default="nvidia/Cosmos-1.0-Autoregressive-4B", type=str, help="The path to the pretrained nemo model"
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="The number of nodes to use")
    parser.add_argument("--devices", default=1, type=int, help="The number of devices to use")
    parser.add_argument("--tp", default=8, type=int, help="Tensor parallel size")
    parser.add_argument("--max_steps", default=10000, type=int, help="The max number of steps to run finetuning")
    parser.add_argument("--val_check_interval", default=100, type=int, help="The number of steps to run validation")
    parser.add_argument("--global_batch_size", default=1, type=int, help="The global batch size")
    parser.add_argument("--micro_batch_size", default=1, type=int, help="The micro batch size")
    parser.add_argument("--lr", default=5e-4, type=float, help="The learning rate")
    parser.add_argument("--latent_shape", default=[5, 40, 64], type=int, nargs=3, help="The latent shape")

    args = parser.parse_args()

    main(args)
