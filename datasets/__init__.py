# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .telecomtower import TelecomTowerDetectionDataset, TelecomTowerPredictionDataset, TelecomTowerDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "telecomtower": [TelecomTowerDetectionDataset, TelecomTowerDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1](args)
    
    dataset_dict = {
        "train": dataset_builder(dataset_config, split_set="train", root_dir=args.dataset_root_dir, augment=False),
        "test": dataset_builder(dataset_config, split_set="val", root_dir=args.dataset_root_dir, augment=False),
    }
    return dataset_dict, dataset_config
    