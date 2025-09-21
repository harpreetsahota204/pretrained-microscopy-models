import os
import re
import logging
from PIL import Image
from typing import Dict, Any, List, Union, Optional 

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
import pretrained_microscopy_models as pmm
import segmentation_models_pytorch as smp
import imageio

import fiftyone as fo
from fiftyone import Model

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Pretrained_Microscopy_Model(Model):
    """A FiftyOne model for running Pretrained Microscopy Model.
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = None,
        **kwargs
    ):
        
        self.model_path = model_path
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Configuration
        self.architecture = 'UnetPlusPlus'
        self.encoder = 'resnet50'
        self.encoder_weights = 'micronet'
        self.classes = 4
        
        # Create the model (initially with no pretrained weights)
        self.model = pmm.segmentation_training.create_segmentation_model(
            architecture=self.architecture,
            encoder=self.encoder,
            encoder_weights=None,  # Don't load weights yet
            classes=self.classes
        )
        
        # Download and load the MicroNet pretrained encoder weights
        url = pmm.util.get_pretrained_microscopynet_url(self.encoder, self.encoder_weights)
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        self.model.encoder.load_state_dict(model_zoo.load_url(url, map_location=map_location))
        
        # IMPORTANT: Set model to evaluation mode BEFORE moving to device
        self.model.eval()
        self.model = self.model.to(self.device)  # Note: assign back to model
        
        # Get the preprocessing function (always use 'imagenet' stats)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, 'imagenet')
        
        logger.info(f"Loading model from {model_path}")

    @property
    def media_type(self):
        return "image"
    
    def _predict(self, image: Image.Image) -> Union[fo.Detections, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/keypoint/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # For large images with automatic patching
        with torch.no_grad():
            predictions = pmm.segmentation_training.segmentation_models_inference(
                image_np, 
                self.model, 
                self.preprocessing_fn,
                device=self.device,
                batch_size=8,
                patch_size=512,
                num_classes=self.classes
            )
        
        # Debug output
        logger.debug(f"Predictions shape: {predictions.shape}")
        logger.debug(f"Any detections: {predictions.any()}")
        for i in range(predictions.shape[2]):
            logger.debug(f"  Class {i+1}: {predictions[..., i].sum()} pixels detected")
        
        # Convert to FiftyOne segmentation format
        # predictions shape: (H, W, num_classes) with values 0 or 1
        mask = np.zeros(predictions.shape[:2], dtype=np.uint8)
        for class_idx in range(self.classes):
            class_mask = predictions[..., class_idx] > 0.5
            mask[class_mask] = class_idx + 1  # Class IDs start from 1
        
        return fo.Segmentation(mask=mask)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image)