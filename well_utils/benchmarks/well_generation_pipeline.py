import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from the_well.benchmark.models import FNO, UNetConvNext, UNetClassic

class WellGenerationPipeline:
    """
    A pipeline for the Well model that mimics the interface of ARBaseGenerationPipeline.
    This allows it to be used as a drop-in replacement in evaluation code.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "UNetConvNext",
        device: str = None,
        **kwargs
    ):
        """
        Initialize the generation pipeline.
        
        Args:
            model_path: Path to the model checkpoint
            model_type: Type of model to load ("FNO" or "UNetConvNext")
            device: Device to use for inference ("cuda" or "cpu")
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load the appropriate model
        if model_type == "UNetConvNext":
            self.model = UNetConvNext.from_pretrained(model_path)
        elif model_type == "FNO":
            self.model = FNO.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def generate(
        self,
        inp_vid: torch.Tensor,
        num_input_frames: int,
        seed: Optional[int] = None,
        sampling_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate future frames based on input frames.
        
        Args:
            inp_vid: Input video tensor of shape [C, T, H, W]
            num_input_frames: Number of input frames to use
            seed: Random seed for reproducibility
            sampling_config: Configuration for sampling (ignored, for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)
            
        Returns:
            Generated video as numpy array of shape [C, T, H, W]
        """
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Ensure input is a tensor
        if not isinstance(inp_vid, torch.Tensor):
            inp_vid = torch.tensor(inp_vid, device=self.device)
        else:
            inp_vid = inp_vid.to(self.device)
            
        if inp_vid.dim() == 4:
            # Assuming [C, T, H, W], add batch dimension
            inp_vid = inp_vid.unsqueeze(0)  # [1, C, T, H, W]
            
        # Extract only the input frames
        input_frames = inp_vid[:, :, :num_input_frames]

        input_frames = input_frames.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            B, C, T, H, W = input_frames.shape
            reshaped_input = input_frames.reshape(B, C * T, H, W)
            output = self.model(reshaped_input)

            
        output = output.unsqueeze(1)
            
        combined = torch.cat([input_frames, output], dim=1)

        combined = combined.permute(0, 2, 1, 3, 4)
        
        return combined.cpu()[0]
