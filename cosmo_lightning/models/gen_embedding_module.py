import os
import torch
from einops import rearrange
from typing import List
from lightning import LightningModule
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from cosmo_lightning.models.universal_multi_decoder_vae_module import UniversalMultiDecoderVAEModule


class GenEmbeddingModule(LightningModule):
    def __init__(
        self,
        vae_module: UniversalMultiDecoderVAEModule,
        output_dir: str,
    ):
        super().__init__()
        
        self.video_tokenizer = vae_module.model
        self.output_dir = output_dir
        
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
        
        self.automatic_optimization = False
        self.video_tokenizer.eval()
        # self.video_tokenizer.requires_grad_(False)
    
    def set_data_metadata(self, dataset_name: str, channel_names: List[str]):
        self.dataset_name = dataset_name
        self.channel_names = channel_names
    
    def embed_step(self, batch, split):
        self.video_tokenizer.eval()
        with torch.no_grad():
            video_batch = batch
            (quant_info, _, _), _ = self.video_tokenizer.encode(
                video_batch, self.dataset_name, self.channel_names
            )
            
            for video_idx in range(quant_info.shape[0]):
                single_video_quant = quant_info[video_idx]  # [T, H, W]
                indices = rearrange(single_video_quant, "T H W -> (T H W)").detach().cpu()
                self.builders_dict[split].add_item(torch.IntTensor(indices))
                self.builders_dict[split].end_document()
    
    def on_train_epoch_start(self):
        builders_train = indexed_dataset.make_builder(
            os.path.join(self.output_dir, "train", f"embeddings_{self.global_rank}.bin"),
            impl="mmap",
            chunk_size=64,
            pad_id=0,
            retrieval_db=None,
            vocab_size=64000,
            stride=64,
        )
        builders_val = indexed_dataset.make_builder(
            os.path.join(self.output_dir, "val", f"embeddings_{self.global_rank}.bin"),
            impl="mmap",
            chunk_size=64,
            pad_id=0,
            retrieval_db=None,
            vocab_size=64000,
            stride=64,
        )
        builders_test = indexed_dataset.make_builder(
            os.path.join(self.output_dir, "test", f"embeddings_{self.global_rank}.bin"),
            impl="mmap",
            chunk_size=64,
            pad_id=0,
            retrieval_db=None,
            vocab_size=64000,
            stride=64,
        )
        self.builders_dict = {
            "train": builders_train,
            "val": builders_val,
            "test": builders_test,
        }
    
    def training_step(self, batch, batch_idx):
        self.embed_step(batch, "train")
    
    def on_train_epoch_end(self):
        self.builders_dict["train"].finalize(
            os.path.join(self.output_dir, "train", f"embeddings_{self.global_rank}.idx")
        )
    
    def validation_step(self, batch, batch_idx):
        self.embed_step(batch, "val")
    
    def on_validation_epoch_end(self):
        self.builders_dict["val"].finalize(
            os.path.join(self.output_dir, "val", f"embeddings_{self.global_rank}.idx")
        )
    
    def test_step(self, batch, batch_idx):
        self.embed_step(batch, "test")
    
    def on_test_epoch_end(self):
        self.builders_dict["test"].finalize(
            os.path.join(self.output_dir, "test", f"embeddings_{self.global_rank}.idx")
        )
    
    def configure_optimizers(self):
        return

# vae_module = UniversalVAEModule(
#     variables=[
#         "tracer",
#         "buoyancy",
#         "pressure",
#         "concentration",
#         "velocity_x",
#         "velocity_y",
#         "D_xx",
#         "D_xy",
#         "D_yx",
#         "D_yy",
#         "E_xx",
#         "E_xy",
#         "E_yx",
#         "E_yy",
#     ],
#     max_video_size=[33, 512, 512],
#     # patcher_type="cross_attn",
#     patcher_type="padded",
#     patch_emb_dim=1024,
#     patch_emb_nheads=16,
#     z_channels=16,
#     z_factor=1,
#     patch_size=4,
#     patch_method="haar",
#     channels=128,
#     channels_mult=[2, 4, 4],
#     num_res_blocks=2,
#     attn_resolutions=[32],
#     dropout=0.0,
#     resolution=1024,
#     spatial_compression=16,
#     temporal_compression=8,
#     legacy_mode=False,
#     levels=[8, 8, 8, 5, 5, 5],
#     embedding_dim=6,
#     pretrained_path='/eagle/MDClimSim/tungnd/physics_sim/cosmos_ckpts/Cosmos-1.0-Tokenizer-DV8x16x16/autoencoder_14c.pt',
# )
