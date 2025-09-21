from .zoo import Pretrained_Microscopy_Model


def download_model(model_name, model_path, **kwargs):
    """Downloads the model (no-op since weights are downloaded on load)."""
    # Model weights are downloaded automatically when loading
    pass


def load_model(model_name, model_path, **kwargs):
    """Loads the model.
    
    Args:
        model_name: the name of the model to load
        model_path: the path to the model directory
        **kwargs: additional arguments including:
            - classes: number of segmentation classes (default: 4)
            - architecture: model architecture (default: 'UnetPlusPlus')
            - encoder: encoder backbone (default: 'resnet50')
            - encoder_weights: pretrained weights (default: 'micronet')
    
    Returns:
        a Pretrained_Microscopy_Model instance
    """
    return Pretrained_Microscopy_Model(model_path=model_path, **kwargs)