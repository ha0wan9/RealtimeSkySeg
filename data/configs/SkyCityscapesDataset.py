import os.path as osp
import numpy as np

from mmseg.datasets import CityscapesDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class SkyCityscapesDataset(CityscapesDataset):
    """Custom Cityscapes dataset with only 'sky' and 'not_sky' classes.
    
    Inherits from the CityscapesDataset class provided by MMSegmentation.
    """
    
    # Define the class names and their corresponding colors for visualization
    METAINFO = dict(
        classes = ('not_sky', 'sky'),
        palette = [[0, 0, 0], [70, 130, 180]]
    )

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the custom dataset class with cityscapes suffixes."""
        super().__init__(*args, **kwargs)
