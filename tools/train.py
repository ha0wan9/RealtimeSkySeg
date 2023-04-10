import os.path as osp
import argparse

import mmcv
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmcv import Config


class SkySegmentationTrainer:
    """A class to train a sky segmentation model using BiSeNetV2 and a custom Cityscapes dataset."""

    def __init__(self, config_path):
        """
        Initialize the trainer with the given configuration file.

        Args:
            config_path (str): Path to the modified configuration file.
        """
        self.cfg = Config.fromfile(config_path)

    def set_seed_and_gpus(self):
        """Set the random seed for reproducibility and configure GPU(s) for training."""
        self.cfg.seed = 0
        set_random_seed(self.cfg.seed, deterministic=False)
        self.cfg.gpu_ids = range(1)

    def build_datasets(self):
        """Build the training dataset."""
        self.datasets = [build_dataset(self.cfg.data.train)]

    def build_model(self):
        """Build the segmentation model."""
        self.model = build_segmentor(self.cfg.model)

    def train(self):
        """Train the segmentation model."""
        train_segmentor(
            self.model,
            self.datasets,
            self.cfg,
            distributed=False,
            validate=True
        )


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a sky segmentation model using BiSeNetV2 and a custom Cityscapes dataset."
    )
    parser.add_argument(
        "config_path",
        help="Path to the modified configuration file."
    )
    return parser.parse_args()


def main():
    """
    Main function to run the training script.
    """
    # Parse the command-line arguments
    # args = parse_args()

    config_path = 'configs/bisenetv2_fcn_4x4-160k_skycityscapes-1024x1024.py'

    # Create a SkySegmentationTrainer instance with the provided configuration file
    trainer = SkySegmentationTrainer(config_path)

    # Set random seed and configure GPU(s)
    trainer.set_seed_and_gpus()

    # Build the training dataset
    trainer.build_datasets()

    # Build the segmentation model
    trainer.build_model()

    # Train the segmentation model
    trainer.train()


if __name__ == "__main__":
    main()

