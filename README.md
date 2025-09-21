# Pretrained Microscopy Models for FiftyOne

This repository provides pretrained microscopy segmentation models as a FiftyOne Model Zoo source. The models use U-Net++ architecture with ResNet50 encoders pretrained on MicroNet datasets for improved microscopy image segmentation.

## Installation

First, install the NASA pretrained microscopy models package:

```bash
pip install git+https://github.com/nasa/pretrained-microscopy-models#egg=pretrained_microscopy_models
```

## Quick Start

### 1. Load a Sample Dataset

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load a microscopy dataset from Hugging Face
dataset = load_from_hub("Voxel51/OD_MetalDAM", overwrite=True)
```

### 2. Register the Model Zoo Source

```python
import fiftyone.zoo as foz

# Register this repository as a model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/pretrained-microscopy-models",
    overwrite=True
)
```

### 3. Load and Apply the Model

```python
# Load the model
model = foz.load_zoo_model(
    "pretrained-microscopy-models/unetplusplus-resnet50-micronet",
    classes=4  # Number of segmentation classes
)

# Apply to dataset
dataset.apply_model(model, label_field="segmentation")

# Visualize results
session = fo.launch_app(dataset)
```

## How It Works

The model uses a **U-Net++ architecture** with a **ResNet50 encoder** that has been pretrained on the MicroNet dataset - a large-scale collection of microscopy images. This pretraining improves performance on microscopy-specific features compared to models pretrained only on natural images.

The inference process:

1. **Preprocessing**: Images are normalized using ImageNet statistics

2. **Patching**: Large images are automatically divided into 512x512 patches for processing

3. **Prediction**: Each patch is processed through the model to generate class probabilities

4. **Reassembly**: Patches are stitched back together to create the full segmentation mask

## Model Parameters

When loading the model, you can customize:

```python
model = foz.load_zoo_model(
    "pretrained-microscopy-models/unetplusplus-resnet50-micronet",
    classes=3,                      # Number of segmentation classes
    architecture='FPN',              # Options: 'Unet', 'UnetPlusPlus', 'FPN'
    encoder='resnet34',              # Any supported encoder
    encoder_weights='image-micronet' # Options: 'micronet', 'image-micronet'
)
```

## Adding Support for Other Models

Currently, this integration supports ResNet50-based models. The [original NASA repository](https://github.com/nasa/pretrained-microscopy-models) includes many more pretrained encoders (DenseNet, EfficientNet, MobileNet, VGG, etc.).

To add support for additional models, simply add new entries to `manifest.json`:

```json
{
    "base_name": "pretrained-microscopy-models/unet-densenet121-micronet",
    "base_filename": "unet-densenet121-micronet",
    "author": "Your Name",
    "license": "Apache-2.0",
    "source": "https://github.com/harpreetsahota204/pretrained-microscopy-models",
    "description": "U-Net with DenseNet121 encoder pretrained on MicroNet",
    "tags": ["segmentation", "microscopy", "densenet121", "pytorch"],
    "date_added": "2025-09-19",
    "requirements": {
        "packages": ["pretrained-microscopy-models", "torch", "torchvision"],
        "cpu": {"support": true},
        "gpu": {"support": true}
    }
}
```

The current implementation in `zoo.py` already supports different architectures and encoders through parameters, so new models will work automatically once added to the manifest.

## Citation

If you use these models in your research, please cite:

```bibtex
@article{norouzzadeh2022micronet,
  title={MicroNet: A unified model for segmentation of various objects in microscopy images},
  author={Norouzzadeh, Mohammadmehdi and Nguyen, Quan and Myles, Brian and Mott, Kevin and Strachan, Jarred},
  journal={npj Computational Materials},
  volume={8},
  number={1},
  pages={200},
  year={2022},
  publisher={Nature Publishing Group},
  doi={10.1038/s41524-022-00878-5}
}
```

## Acknowledgments

This FiftyOne integration is based on the original [NASA Pretrained Microscopy Models](https://github.com/nasa/pretrained-microscopy-models) repository, which provides the core pretrained encoders and model architectures.

## License

Apache-2.0