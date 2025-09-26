import logging

from torch.utils.data import DataLoader
from utils.dataset_loading import parseDataset, parseDatasetFromJSON
from dataset import (
    AugmentedGeneralDataset,
    GeneralDataset,
    OSM_CLIP_Dataset,
    custom_collate_fn_RSICD,
)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

DATASET_PATH = "/media/Melgani/Riccardo/Datasets/image_captioning/RSICD_captions"
DATASET_PATH_UCM = "/media/Melgani/Riccardo/Datasets/image_captioning/UCM_captions"
DATASET_PATH_SIDNEY = (
    "/media/Melgani/Riccardo/Datasets/image_captioning/SIDNEY_Captions"
)
DATASET_PATH_NWPU = (
    "/media/Melgani/Riccardo/Datasets/image_captioning/NWPU_Captions/dataset_nwpu.json"
)


def load_datasets_as_single_one(
    datasetsToUse, BATCH_SIZE, preprocess, aug_new_images=False, num_of_aug_images=0
):
    train_datasets = []
    val_datasets = []
    test_datasets = []

    if "RSICD" in datasetsToUse:
        train_NWPU, val_NWPU, test_NWPU = parseDatasetFromJSON(
            DATASET_PATH_NWPU, "NWPU", ".jpg"
        )
        train_datasets.extend(train_NWPU)
        val_datasets.extend(val_NWPU)
        test_datasets.extend(test_NWPU)

    if "UCM" in datasetsToUse:
        train_UCM, val_UCM, test_UCM = parseDataset(DATASET_PATH_UCM, "UCM", ".tif")
        train_datasets.extend(train_UCM)
        val_datasets.extend(val_UCM)
        test_datasets.extend(test_UCM)

    if "SIDNEY" in datasetsToUse:
        train_SIDNEY, val_SIDNEY, test_SIDNEY = parseDataset(
            DATASET_PATH_SIDNEY, "SIDNEY", ".tif"
        )
        train_datasets.extend(train_SIDNEY)
        val_datasets.extend(val_SIDNEY)
        test_datasets.extend(test_SIDNEY)

    if "NWPU" in datasetsToUse:
        train_RSICD, val_RSICD, test_RSICD = parseDataset(DATASET_PATH, "RSICD", ".jpg")
        train_datasets.extend(train_RSICD)
        val_datasets.extend(val_RSICD)
        test_datasets.extend(test_RSICD)

    # Create combined dataset objects
    train_datasets_after_aug = AugmentedGeneralDataset(
        train_datasets, preprocess, aug_new_images, num_of_aug_images
    )
    val_dataset = GeneralDataset(val_datasets, preprocess)
    test_dataset = GeneralDataset(test_datasets, preprocess)

    # Log the dataset sizes including augmentations
    original_train_size = len(train_datasets)
    augmented_train_size = len(train_datasets_after_aug)
    logging.info(f"Original training samples: {original_train_size}")
    logging.info(f"Training samples after augmentation: {augmented_train_size}")
    if aug_new_images:logging.info(f"Added {augmented_train_size - original_train_size} augmented samples")
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_datasets_after_aug,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn_RSICD,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn_RSICD,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn_RSICD,
    )

    print(f"Original dataset size: {len(train_datasets)}")
    print(f"Augmented dataset size: {len(train_datasets_after_aug)}")
    total_samples = len(train_dataloader.dataset)
    print(f"Total samples in DataLoader: {total_samples}")
    return train_dataloader, val_dataloader, test_dataloader


def load_osm_dataset(
    dataset_path: str,
    batch_size: int,
    image_size: int,
    train_percentage: float,
    val_percentage: float,
    preprocess: callable,
    patch_size: int,
    collate_fn: callable,
    json_path: str,
    mask_path: str,
):
    dataset = OSM_CLIP_Dataset(
        dataset_path,
        preprocess,
        patch_size=patch_size,
        image_size=image_size,
        json_path=json_path,
        mask_path=mask_path,
    )

    # Define the sizes for train, validation, and test sets (e.g., 70%, 20%, 10%)
    train_size = int(train_percentage * len(dataset))
    val_size = int(val_percentage * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoader instances for each set
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, image_size, patch_size),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, image_size, patch_size),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, image_size, patch_size),
    )

    return train_dataloader, val_dataloader, test_dataloader
