import argparse
import json
import os
import logging
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional
import random
import imageio
import torch
import omegaconf
from glob import glob
from cosmos1.models.autoregressive.inference.custom_world_generation_pipeline import CustomARBaseGenerationPipeline
from cosmos1.models.autoregressive.utils.inference import add_common_arguments, load_vision_input, validate_args, prepare_array_batch, NUM_TOTAL_FRAMES
from cosmos1.models.autoregressive.utils.evaluation2 import HDF5DataLoader
from cosmos1.utils import log
from well_utils.data_processing.normalization.normalize import NormalizationApplier 
from well_utils.data_processing.visualizations import create_video_heatmap
from well_utils.metrics.spatial import MSE, VRMSE
from cosmo_lightning.models.refinement_mdoule import RefinementModule

CHANNEL_NAMES = None #['tracer', 'pressure', 'velocity_x', 'velocity_y']

IGNORE_CHANNELS = [
    'mask_HS',
    'density_ASD', 'density_ASI', 'density_ASM',
    'speed_of_sound_ASD', 'speed_of_sound_ASI', 'speed_of_sound_ASM',
]

class VideoEvaluator:
    """Main class for handling video evaluation pipeline."""
    
    def __init__(self, args: argparse.Namespace, eval_indices: List[int] = None, 
                 random_eval_samples: int = None):
        self.args = args
        self.eval_indices = self.args.eval_indices
        self.random_eval_samples = self.args.random_eval_samples
        self.normalizer = self._load_normalizer()
        self.pipeline = self._initialize_pipeline()
        self.sampling_config = validate_args(args, "base")
        self.metrics = {
            "MSE": MSE(n_spatial_dims=2, reduce=True),
            "VRMSE": VRMSE(n_spatial_dims=2, reduce=True)
        }
        self.num_runs_per_sample = self.args.num_runs_per_sample
        self.num_rollouts = self.args.num_rollouts
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.refiner_model = self._initialize_refiner()
        os.makedirs(self.args.output_dir, exist_ok=True)
        # Create a directory for saving numpy arrays
        self.arrays_dir = os.path.join(self.args.output_dir, "arrays")
        os.makedirs(self.arrays_dir, exist_ok=True)

    def _calculate_frame_metrics(self, pred_frames: np.ndarray, gt_frames: np.ndarray) -> List[Dict[str, float]]:
        """Calculate metrics for each frame in the sequence."""
        frame_metrics = []
        min_length = min(pred_frames.shape[1], gt_frames.shape[1])
        
        for frame_idx in range(min_length):
            frame_metrics.append({
                metric: self.metrics[metric](
                    pred_frames[:, frame_idx:frame_idx+1],
                    gt_frames[:, frame_idx:frame_idx+1]
                ).item()
                for metric in self.metrics
            })
        return frame_metrics

        
    def _load_normalizer(self) -> NormalizationApplier:
        """Load normalization statistics from JSON file."""
        return NormalizationApplier(self.args.channel_stats_path, normalization_type=self.args.normalization_type)

    def _load_data(self) -> HDF5DataLoader:
        """Initialize HDF5 data loader with configured parameters"""
        # Calculate total frames needed for rollouts
        if not self.args.total_frames:
            total_frames_needed = NUM_TOTAL_FRAMES
            total_frames_needed += self.num_rollouts * (NUM_TOTAL_FRAMES - self.args.context_len)
        else:
            total_frames_needed = self.args.total_frames
            
        return HDF5DataLoader(
            path=self.args.batch_input_path,
            num_input_frames=self.args.num_input_frames,
            num_total_frames=total_frames_needed,
            data_resolution=self.args.dimensions,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )     
    
    def _visualize_sample(self, gt_array, pred_array, filename):
        # visualized_video = np.concatenate([gt_array.transpose(1, 2, 3, 0), pred_array.transpose(1, 2, 3, 0)], axis=1)
        # visualized_video = np.clip(visualized_video, 0, 255).astype(np.uint8)
        # imageio.mimsave(f"evaluation_visualizations/{filename}.mp4", visualized_video, fps=20)

        concat_videos = np.stack([gt_array, pred_array], axis=0)
        create_video_heatmap(concat_videos, save_path=f"{self.args.output_dir}/{filename}.mp4", fps=20, channel_names=CHANNEL_NAMES)
    
    def _initialize_pipeline(self) -> CustomARBaseGenerationPipeline:
        """Initialize the generation pipeline with configured parameters."""
        config_path = os.path.join(self.args.tokenizer_path, "config.yaml")
        config = omegaconf.OmegaConf.load(config_path)
        
        ckpt_paths = glob(os.path.join(self.args.tokenizer_path, "checkpoints", "epoch_*.ckpt"))
        assert len(ckpt_paths) == 1, f"There should be only one best checkpoint in {self.args.tokenizer_path}/checkpoints"
        ckpt_path = ckpt_paths[0]
        
        dataset_name = None
        for name in config['data']['metadata_dict'].keys():
            if name in self.args.batch_input_path:
                dataset_name = name
                break
        channel_names = config['data']['metadata_dict'][dataset_name]['channel_names']
        
        return CustomARBaseGenerationPipeline(
            inference_type="base",
            checkpoint_dir=self.args.checkpoint_dir,
            checkpoint_name=self.args.ar_model_dir,
            tokenizer_path=ckpt_path,
            dataset_name=dataset_name,
            channel_names=channel_names,
            omega_conf=config,
            # max_latent_shape=self.args.max_latent_shape,
            # dataset_latent_shape=[
            #     self.args.max_latent_shape[0],
            #     self.args.dimensions[0] // self.args.compression_ratio[1],
            #     self.args.dimensions[1] // self.args.compression_ratio[2],
            # ],
            disable_diffusion_decoder=self.args.disable_diffusion_decoder,
            offload_guardrail_models=self.args.offload_guardrail_models,
            offload_diffusion_decoder=self.args.offload_diffusion_decoder,
            offload_network=self.args.offload_ar_model,
            offload_tokenizer=self.args.offload_tokenizer,
            overwrite_ckpt_path=self.args.overwrite_ckpt_path,
            video_height=self.args.dimensions[0],
            video_width=self.args.dimensions[1],
            compression_ratio=self.args.compression_ratio,
            enable_video_guardrail=False,
            context_len=self.args.context_len,
            ignore_first_latent=self.args.ignore_first_latent,
        )
    
    def _initialize_refiner(self) -> Optional[RefinementModule]:
        """Initialize the refinement model if checkpoint path is provided."""
        if self.args.refiner_checkpoint_path:
            log.info(f"Loading RefinementModule from: {self.args.refiner_checkpoint_path}")
            try:
                refiner_model = RefinementModule.load_from_checkpoint(
                    self.args.refiner_checkpoint_path, 
                    map_location='cpu' # Load on CPU first
                )
                refiner_model.to(self.device, dtype=torch.float32) # Move to the target device
                refiner_model.eval() # Set to evaluation mode
                log.info("RefinementModule loaded successfully.")
                return refiner_model
            except FileNotFoundError:
                log.error(f"Refiner checkpoint not found at {self.args.refiner_checkpoint_path}. Refiner disabled.")
                return None
            except Exception as e:
                log.error(f"Error loading refiner model: {e}. Refiner disabled.")
                return None
        else:
            log.info("No refiner checkpoint path provided. Refiner disabled.")
            return None

    def _get_evaluation_indices(self, num_samples: int) -> List[int]:
        """Determine which indices to evaluate based on initialization parameters"""
        if self.eval_indices:
            return [i for i in self.eval_indices if i < num_samples]
        if self.random_eval_samples:
            return np.random.choice(num_samples, size=min(num_samples, self.random_eval_samples), replace=False).tolist()
        return list(range(num_samples))

    @torch.no_grad()
    def refine_generated_sequence(self, pred_frames, ground_truth):
        """Refine a sequence of frames using the same approach as during training."""
        if self.refiner_model is None:
            return pred_frames
        
        refiner_context_frames = self.refiner_model.hparams.context_frames
        refiner_grounding_indices = self.refiner_model.hparams.gounding_frames # or grounding_frames
        b, c_out, t_total, h, w = pred_frames.shape
        
        refined_sequence = torch.zeros((b, c_out, t_total, h, w)).to('cpu')
        refined_sequence[:, :, :refiner_context_frames] = ground_truth[:, :, :refiner_context_frames]
        
        context = ground_truth.clone().to('cpu')
        for target_idx in range(refiner_context_frames, min(13, t_total)):
            
            grounding_idx = [target_idx + i for i in refiner_grounding_indices]
            grounding_frames = pred_frames[:,:,grounding_idx]  # Extra dimension with [None]
            # breakpoint()
            refiner_input = torch.cat([context, grounding_frames], dim=2)
            # breakpoint()
            refiner_input = refiner_input.flatten(1,2)  #( N C T_CONTEXT + T_GROUNDING )H W

            
            target_idx_tensor = torch.tensor([target_idx] * b, device=self.device, dtype=torch.long)
            
            refined_frame = self.refiner_model.model(
                refiner_input.to(self.device), 
                target_idx_tensor
            )
            refined_frame = refined_frame.to('cpu')
            
            refined_sequence[:, :, target_idx, :, :] = refined_frame
        
        return refined_sequence

    def _perform_rollout_prediction(self, input_frames, ground_truth, run_seed=None):
        """Perform prediction with multiple rollouts"""
        context_len = self.args.context_len
        all_predictions = []

        refiner_gt = ground_truth[:, :, :context_len]
        
        # Initial prediction using the first frames
        current_input = input_frames[:, :, :NUM_TOTAL_FRAMES]
        pred_frames = self.pipeline.generate(
            inp_vid=current_input,
            num_input_frames=self.args.num_input_frames,
            seed=run_seed,
            sampling_config=self.sampling_config,
        )

        if pred_frames is None:
            log.warning("Initial video generation failed")
            return None
        
        pred_frames = torch.tensor(pred_frames).unsqueeze(0)
        # pred_frames[:, :, :context_len] = input_frames[:, :, :context_len]
        
        # Refine the initial prediction as a 13-frame window
        if self.refiner_model is not None:
            log.info("Refining initial prediction frames...")
            pred_frames = self.refine_generated_sequence(pred_frames, input_frames[:, :, :context_len])
            log.info("Initial prediction frames refined.")

        all_predictions.append(pred_frames)
        
        # For each rollout, use the last context_len frames from previous prediction
        # But still maintain the 13-frame window structure for refinement
        for rollout in range(self.num_rollouts):
            # Get the last context_len frames from previous prediction
            last_context = pred_frames[:, :, -context_len:]
            
            # Generate a new 13-frame sequence
            next_input = torch.cat((
                last_context, 
                torch.zeros(1, last_context.shape[1], NUM_TOTAL_FRAMES-context_len, 
                          last_context.shape[3], last_context.shape[4])
            ), dim=2)
            
            pred_frames = self.pipeline.generate(
                inp_vid=next_input,
                num_input_frames=context_len,
                seed=run_seed,
                sampling_config=self.sampling_config,
            )
            
            if pred_frames is None:
                log.warning(f"Video generation failed for rollout {rollout+1}, stopping rollouts")
                break
            
            pred_frames = torch.tensor(pred_frames).unsqueeze(0)
            # pred_frames[:, :, :context_len] = last_context
            
            if self.refiner_model is not None:
                log.info(f"Refining frames for rollout {rollout+1}...")
                pred_frames = self.refine_generated_sequence(pred_frames, last_context)
                log.info(f"Rollout {rollout+1} frames refined.")
            
            all_predictions.append(pred_frames[:, :, context_len:])
        
        # Concatenate all predictions
        first_pred = all_predictions[0]
        full_prediction = torch.cat([first_pred] + [pred for pred in all_predictions[1:]], axis=2)
        
        return full_prediction
    
    def _process_single_video(self, input_frames: torch.Tensor, ground_truth: torch.Tensor, filename = None, sample_idx=0) -> Dict[str, float]:
        """Process a single video and calculate metrics."""
        # Generate predictions
        print("Input frames shape:", input_frames.shape, "Max:", input_frames.max(), "Min:", input_frames.min(), "Mean:", input_frames.mean())
        print("Ground truth shape:", ground_truth.shape, "Max:", ground_truth.max(), "Min:", ground_truth.min(), "Mean:", ground_truth.mean())
        
        # Run predictions and average them
        all_predictions = []
        
        for run in range(self.num_runs_per_sample):
            run_seed = self.args.seed + run if self.args.seed is not None else None
            
            pred_frames = self._perform_rollout_prediction(input_frames, ground_truth, run_seed)
            
            if pred_frames is None:
                log.warning(f"Video generation failed for run {run}, skipping")
                continue
                
            all_predictions.append(pred_frames.to(torch.float32))
        
        if not all_predictions:
            raise RuntimeError("All video generation attempts failed")
            
        # Average the predictions (even if there's just one)
        ground_truth = ground_truth.to(torch.float32)
        predicted_frames = np.mean(np.stack(all_predictions, axis=0), axis=0)
        if self.num_runs_per_sample > 1:
            log.info(f"Created average prediction from {len(all_predictions)} successful runs")
                
        # Convert tensors to numpy arrays and transpose to channels-last format
        pred_np = predicted_frames[0][:, :self.args.total_frames]
        gt_np = prepare_array_batch(ground_truth)[0][:, :self.args.total_frames]
        
        # Calculate how many frames we should use for metrics
        # Initial prediction (NUM_TOTAL_FRAMES) + rollout predictions
        total_pred_length = NUM_TOTAL_FRAMES + self.num_rollouts * (NUM_TOTAL_FRAMES - self.args.context_len)
        
        # Make sure we don't go beyond what we've predicted or the ground truth
        total_pred_length = min(total_pred_length, pred_np.shape[1], gt_np.shape[1])

        pred_denorm = self.normalizer.inverse_norm(pred_np.transpose(1, 2, 3, 0)).transpose(3, 0, 1, 2)
        gt_denorm = self.normalizer.inverse_norm(gt_np.transpose(1, 2, 3, 0)).transpose(3, 0, 1, 2)

        # Save numpy arrays for later evaluation in a compressed format
        np.savez_compressed(os.path.join(self.arrays_dir, f"sample_{sample_idx}.npz"), pred=pred_denorm, gt=gt_denorm)
        
        if filename is not None:
            try:
                self._visualize_sample(gt_denorm, pred_denorm, filename)
                pass
                # self.visualize_videos([gt_denorm[:min_length], pred_denorm[:min_length]], save_path=f"evaluation_visualizations/{filename}.mp4")
            except Exception as e:
                log.error(f"Error visualizing videos: {e}")
        
        # ignore constant channels
        valid_indices = [i for i, name in enumerate(self.pipeline.inference_config.tokenizer_config.video_tokenizer.channel_names) if name not in IGNORE_CHANNELS]
        pred_denorm = pred_denorm[valid_indices]
        gt_denorm = gt_denorm[valid_indices]

        print("Data type of pred_denorm:", pred_denorm.dtype)
        print("Data type of gt_denorm:", gt_denorm.dtype)
        frame_metrics = self._calculate_frame_metrics(pred_denorm, gt_denorm)

        # mean metrics
        def print_metrics(metrics):
            for metric in metrics[-1].keys():
                print(f"{metric}: {np.mean([metrics[i][metric] for i in range(len(metrics))])}")
        print_metrics(frame_metrics)

        # If refinement was used, add a note to the log
        if self.refiner_model is not None:
            log.info("Metrics are for predictions with refinement already applied")

        return frame_metrics
    
    def evaluate(self) -> Dict[int, Dict[str, float]]:
        """Main evaluation loop with per-frame metric aggregation."""
        data_loader = self._load_data()
        all_metrics = []
        sample_idx = 0
        for file_samples in data_loader.get_generators():
            num_samples = len(file_samples)
            eval_indices = self._get_evaluation_indices(num_samples)
            
            for idx in eval_indices:
                # try:
                    input_tensor, target_tensor = file_samples[idx]
                    frame_metrics = self._process_single_video(input_tensor, target_tensor, 
                                                              filename=f"sample_{sample_idx%20}",
                                                              sample_idx=sample_idx)
                    all_metrics.append(frame_metrics)
                    sample_idx += 1
                    
                # except Exception as e:
                #     log.error(f"Error processing sample {idx}: {str(e)}")
            # if sample_idx > 100:
            #     break

        with open(f"{self.args.output_dir}/all_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        # Aggregate metrics across samples
        averaged_metrics = {}
        max_frames = max(len(sample) for sample in all_metrics)
        
        for frame_idx in range(max_frames):
            frame_values = defaultdict(list)
            for sample in all_metrics:
                if frame_idx < len(sample):
                    for metric, value in sample[frame_idx].items():
                        frame_values[metric].append(value)
            
            averaged_metrics[frame_idx] = {
                metric: np.mean(values) 
                for metric, values in frame_values.items()
            }
        
                
        return averaged_metrics




def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch video evaluation for generation model")
    add_common_arguments(parser)
    parser.add_argument("--ar_model_dir", type=str, default="Cosmos-Predict1-4B")
    parser.add_argument("--overwrite_ckpt_path", type=str, default="/eagle/MDClimSim/tungnd/physics_sim/universal_vae_8_datasets_balanced_training_cutoff_rope_32_bs_1e-3_lr_5000_warmup_1_epoch/checkpoints/epoch=0-step=10499-val_loss=3.36/converted_model_hf.bin")
    parser.add_argument("--tokenizer_path", type=str, default="/eagle/MDClimSim/tungnd/physics_sim/universal_discrete_vae_all_datasets_separate_channels_separate_decoder_13_frames_8s_4t_32_bs_1e-3_lr_1000_epochs")
    parser.add_argument("--input_type", type=str, default="array", choices=["image", "video", "array"])
    # parser.add_argument("--max_latent_shape", default=[4, 128, 64], type=int, nargs=3, help="The max latent shape across datasets")
    parser.add_argument("--dimensions", type=int, nargs=2, default=[256, 256], 
                       help="Spatial dimensions of the videos (height width)")
    parser.add_argument("--channel_stats_path", type=str, required=True,
                       help="Path to JSON file containing channel normalization statistics")
    parser.add_argument("--normalization_type", type=str, default="standard", choices=["standard", "minmax"])
    parser.add_argument("--eval_indices", type=int, nargs='+', default=None, help="Indices of samples to evaluate")
    parser.add_argument("--random_eval_samples", type=int, default=1, help="Number of random samples to evaluate")
    parser.add_argument("--visualize_interval", type=int, default=-1, help="Interval between videos to visualize")
    parser.add_argument("--compression_ratio", type=int, nargs=3, default=[4, 8, 8], help="Compression ratio for the video")
    parser.add_argument("--output_dir", type=str, default="evaluation_visualizations", help="Output directory for visualizations")
    parser.add_argument("--num_runs_per_sample", type=int, default=1, help="Number of runs per sample to average over")
    parser.add_argument("--context_len", type=int, default=5, help="Context length")
    parser.add_argument("--ignore_first_latent", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=0, help="Number of rollout iterations to perform")
    parser.add_argument("--total_frames", type=int, default=None, help="")
    parser.add_argument("--refiner_checkpoint_path", type=str, default=None, 
                       help="Optional path to the refinement model checkpoint.")
    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()
    evaluator = VideoEvaluator(args)
    
    log.info("Starting evaluation...")
    start_time = time.time()
    metrics = evaluator.evaluate()

    print("Averaged metrics per frame:")
    for frame_idx, metric in metrics.items():
        print(f"Frame {frame_idx}:")
        for metric, value in metric.items():
            print(f"  {metric}: {value:.4f}")

    import json
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Total processing time: ", time.time() - start_time)



if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    main()