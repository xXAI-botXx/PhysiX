import os
import re
from datetime import datetime
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from cosmo_lightning.data.multi_dataset_datamodule import MultiDatasetDataModule
from cosmo_lightning.models.universal_vae_module import UniversalVAEModule
from cosmo_lightning.models.refinement_mdoule import RefinementModule
from cosmo_lightning.data.refinement_data_module import RefinementDataModule
from lightning.pytorch.loggers.wandb import WandbLogger


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


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=RefinementModule,
        datamodule_class=RefinementDataModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    
    logger_name = cli.trainer.logger._name
    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints'),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=cli.trainer.callbacks[i].save_top_k,
                save_last=cli.trainer.callbacks[i].save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[i].auto_insert_metric_name
            )
            
    latest_run_id = extract_latest_run_id(cli.trainer.default_root_dir, logger_name)
    if latest_run_id is not None:
        print("Resuming wandb run with ID:", latest_run_id)
        cli.trainer.logger = WandbLogger(
            name=logger_name,
            project=cli.trainer.logger._wandb_init['project'],
            save_dir=os.path.join(cli.trainer.default_root_dir, logger_name),
            id=latest_run_id,
            resume='must'
        )
    else:
        cli.trainer.logger = WandbLogger(
            name=logger_name,
            project=cli.trainer.logger._wandb_init['project'],
            save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
        )

    if os.path.exists(os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints', 'last.ckpt')):
        ckpt_resume_path = os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints', 'last.ckpt')
    else:
        ckpt_resume_path = None

    # fit() runs the training
    if os.environ.get('EVAL_ONLY'):
        cli.trainer.validate(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)
    else:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)
    

if __name__ == "__main__":
    main()
