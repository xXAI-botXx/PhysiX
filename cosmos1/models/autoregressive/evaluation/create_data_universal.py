import argparse
import os
import time
import numpy as np
from typing import Dict, List
import torch
from einops import rearrange
import omegaconf
from glob import glob

from cosmos1.models.autoregressive.inference.custom_world_generation_pipeline import CustomARBaseGenerationPipeline
from cosmos1.models.autoregressive.utils.inference import add_common_arguments, validate_args, NUM_TOTAL_FRAMES
from cosmos1.models.autoregressive.utils.evaluation2 import HDF5DataLoader
from cosmos1.utils import log
from well_utils.data_processing.normalization.normalize import NormalizationApplier 
from well_utils.data_processing.visualizations import create_video_heatmap
from well_utils.metrics.spatial import MSE, VRMSE
from cosmos1.utils import misc

CHANNEL_NAMES = None #['tracer', 'pressure', 'velocity_x', 'velocity_y']


class VideoEvaluator:
    """Main class for handling video evaluation pipeline."""
    
    def __init__(self, args: argparse.Namespace, eval_indices: List[int] = None, 
                 random_eval_samples: int = None):
        self.args = args
        self.world_size  = int(os.environ['WORLD_SIZE'])
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES']=devices[self.local_rank]
        os.environ['WORLD_SIZE']='1'
        os.environ['LOCAL_RANK']='0'
        print("HHHH",os.environ['CUDA_VISIBLE_DEVICES'])
        torch.cuda.set_device(self.local_rank )
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
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # self.world_size = os.environ['World']
        

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
        return HDF5DataLoader(
            path=self.args.batch_input_path,
            num_input_frames=self.args.num_input_frames,
            num_total_frames=NUM_TOTAL_FRAMES,
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
            context_len=self.args.context_len
        )
    
    def _get_evaluation_indices(self, num_samples: int) -> List[int]:
        """Determine which indices to evaluate based on initialization parameters"""
        if self.eval_indices:
            return [i for i in self.eval_indices if i < num_samples]
        if self.random_eval_samples:
            return np.random.choice(num_samples, size=self.random_eval_samples, replace=False).tolist()
        return list(range(num_samples))

    @torch.no_grad()
    def _process_single_video(self, input_frames: torch.Tensor, ground_truth: torch.Tensor, filename = None) -> Dict[str, float]:
        """Process a single video and calculate metrics."""
        # Generate predictions
        print("Input frames shape:", input_frames.shape, "Max:", input_frames.max(), "Min:", input_frames.min(), "Mean:", input_frames.mean())
        print("Ground truth shape:", ground_truth.shape, "Max:", ground_truth.max(), "Min:", ground_truth.min(), "Mean:", ground_truth.mean())
        all_predictions = []
        output_target = f"{self.args.output_dir}/{filename}.npz"
        # if os.path.exists(output_target):
        #     fp = np.load(output_target)
        #     loaded_data = dict(fp)
        #     fp.close()
        #     del fp
        #     input_frames_load = loaded_data['input_frames']
        #     # breakpoint()
        #     [np.all(input_frames_load[:,x] == input_frames[:,x].cpu().numpy()) for x in range(input_frames.shape[1])]
        #     assert np.all(input_frames_load == input_frames.cpu().numpy())
        #     loaded_data['ground_truth'] = ground_truth.cpu().numpy()
        #     np.savez(output_target, **loaded_data)
        #     # fp.close()
        #     print("SKIPPING")
            
        
        inp_vid = ground_truth
        data_batch = {}
        data_batch["video"] = inp_vid
        data_batch = misc.to(data_batch, "cuda")
        data_tokens, token_boundaries = self.pipeline.model.tokenizer.tokenize(data_batch=data_batch)
        num_eov_tokens = 1 if self.pipeline.model.tokenizer.tokenizer_config.add_special_tokens else 0
        num_bov_tokens = 1 if self.pipeline.model.tokenizer.tokenizer_config.add_special_tokens else 0
        T,H,W = self.pipeline.latent_shape
        latent_context_t_size=2
        # num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))
        # num_tokens_to_generate = num_gen_tokens
        # sample_num = 0
        # input_tokens = data_tokens[sample_num][0 : token_boundaries["video"][sample_num][1]]  # [B, L]
        # input_tokens = [
        #     input_tokens[0 : -num_tokens_to_generate - num_eov_tokens].tolist()
        # ]  # -1 is to exclude eov token
        # breakpoint()
        input_tokens = data_tokens.cuda()
        prompt_len = data_tokens.shape[-1]
        input_pos = torch.arange(0, prompt_len, device="cuda")
        # breakpoint()
        y = self.pipeline.model.model(
            tokens=input_tokens,
            input_pos=input_pos,
        )
        # y.shape
        argmax_prediction = y.argmax(-1)
        argmax_prediction = torch.cat([input_tokens[:,:1], argmax_prediction[:,1:]], dim=1)
        # (input_tokens - argmax_prediction).mean()

        indices_tensor = rearrange(
            argmax_prediction,
            "B (T H W) -> B T H W",
            T=T,
            H=H,
            W=W,
        )
        self.pipeline.model.tokenizer.video_tokenizer.to(indices_tensor.device)
        quant_codes = self.pipeline.model.tokenizer.video_tokenizer.quantizer.indices_to_codes(indices_tensor)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            video_decoded = self.pipeline.model.tokenizer.video_tokenizer.decode(quant_codes, self.pipeline.model.tokenizer.tokenizer_config.video_tokenizer.dataset_name)
            var_ids = self.pipeline.model.tokenizer.video_tokenizer.encoder.patcher3d.get_var_ids(tuple(self.pipeline.model.tokenizer.tokenizer_config.video_tokenizer.channel_names), video_decoded.device)
            video_decoded = video_decoded[:, var_ids]
        # N C T H W
        # breakpoint()
        
        if self.local_rank == 1:
            has_nan = torch.isnan(indices_tensor).any(),torch.isnan(quant_codes).any()
            print(has_nan)
            assert not has_nan[0] and not has_nan[1]
        assert not torch.isnan(video_decoded).any(),f"Local rank: {self.local_rank}, cannot detokenize"
        #[self.metrics['VRMSE'](video_decoded[:,:,i], inp_vid[:,:,i]).item() for i in range(video_decoded.shape[2])]
        pred_frames = self.pipeline.generate(
                inp_vid=input_frames,
                num_input_frames=self.args.num_input_frames,
                seed=0,
                sampling_config=self.sampling_config,
        )
        #[self.metrics['VRMSE'](pred_frames[None][:,:,i], inp_vid[:,:,i].cpu().numpy()).item() for i in range(video_decoded.shape[2])]

        payload =  dict(input_frames=input_frames.cpu().numpy(),
                    quant_codes = quant_codes.float().cpu().numpy(),
                    video_decoded=video_decoded.float().cpu().numpy(),
                    pred_frames=pred_frames,
                    ground_truth=ground_truth.cpu().numpy(),
                    )
        np.savez(output_target, **payload)
        return None
    
    def evaluate(self) -> Dict[int, Dict[str, float]]:
        """Main evaluation loop with per-frame metric aggregation."""
        data_loader = self._load_data()
        # breakpoint()
        all_metrics = []
        i = 0
        for file_samples in data_loader.get_generators():
            num_samples = len(file_samples)
            eval_indices = self._get_evaluation_indices(num_samples)
            #eval_indices = list(range(0,len(file_samples),NUM_TOTAL_FRAMES))
            # file_samples = [0,13,]
            for idx in eval_indices:
                # try:
                    i += 1
                    filename=f"sample_{i}"
                    if i % self.world_size == self.local_rank:
                        input_tensor, target_tensor = file_samples[idx]
                        output_target = f"{self.args.output_dir}/{filename}.npz"
                        if os.path.exists(output_target):
                            continue
                        frame_metrics = self._process_single_video(input_tensor, target_tensor, filename=f"sample_{i}")
                    else:
                        pass
                    # print(filename)
                    # all_metrics.append(frame_metrics)
                    
                # except Exception as e:
                #     log.error(f"Error processing sample {idx}: {str(e)}")
            
            # if i > 100:
            #     break

        # with open(f"{self.args.output_dir}/all_metrics.json", "w") as f:
        #     json.dump(all_metrics, f)
        # Aggregate metrics across samples
        # averaged_metrics = {}
        # max_frames = max(len(sample) for sample in all_metrics)
        
        # for frame_idx in range(max_frames):
        #     frame_values = defaultdict(list)
        #     for sample in all_metrics:
        #         if frame_idx < len(sample):
        #             for metric, value in sample[frame_idx].items():
        #                 frame_values[metric].append(value)
            
        #     averaged_metrics[frame_idx] = {
        #         metric: np.mean(values) 
        #         for metric, values in frame_values.items()
        #     }
        
                
        # return averaged_metrics
        return None




def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch video evaluation for generation model")
    add_common_arguments(parser)
    parser.add_argument("--ar_model_dir", type=str, default="Cosmos-1.0-Autoregressive-4B")
    parser.add_argument("--overwrite_ckpt_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--input_type", type=str, default="video", choices=["image", "video", "array"])
    parser.add_argument("--dimensions", type=int, nargs=2, default=[64, 64], 
                       help="Spatial dimensions of the videos (height width)")
    parser.add_argument("--channel_stats_path", type=str, required=True,
                       help="Path to JSON file containing channel normalization statistics")
    parser.add_argument("--normalization_type", type=str, default="standard", choices=["standard", "minmax"])
    parser.add_argument("--eval_indices", type=int, nargs='+', default=None, help="Indices of samples to evaluate")
    parser.add_argument("--random_eval_samples", type=int, default=None, help="Number of random samples to evaluate")
    parser.add_argument("--visualize_interval", type=int, default=-1, help="Interval between videos to visualize")
    parser.add_argument("--compression_ratio", type=int, nargs=3, default=[8, 16, 16], help="Compression ratio for the video")
    parser.add_argument("--output_dir", type=str, default="evaluation_visualizations", help="Output directory for visualizations")
    parser.add_argument("--num_runs_per_sample", type=int, default=1, help="Number of runs per sample to average over")
    parser.add_argument("--context_len", type=int, default=9, help="Context length")
    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()
    evaluator = VideoEvaluator(args)
    
    log.info("Starting evaluation...")
    start_time = time.time()
    metrics = evaluator.evaluate()

    # print("Averaged metrics per frame:")
    # for frame_idx, metric in metrics.items():
    #     print(f"Frame {frame_idx}:")
    #     for metric, value in metric.items():
    #         print(f"  {metric}: {value:.4f}")

    # import json
    # # with open(f"{args.output_dir}/metrics.json", "w") as f:
    # #     json.dump(metrics, f)

    # print("Total processing time: ", time.time() - start_time)



if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    main()