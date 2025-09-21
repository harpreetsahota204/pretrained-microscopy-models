from .zoo import Pretrained_Microscopy_Model


def download_model(model_name, model_path, **kwargs):
    """Downloads the model (no-op since weights are downloaded on load)."""
    # Model weights are downloaded automatically when loading
    pass


def load_model(model_name, model_path, **kwargs):
    """Loads the model."""
    return Pretrained_Microscopy_Model(model_path=model_path, **kwargs)