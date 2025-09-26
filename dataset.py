import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from retrieves_ids import extract_masks_from_osm_image, visualize_grid_masks
from torchvision.transforms import ToTensor
import clip
from utils.general_utils import prepare_tags
import json
import logging
from torchvision import transforms


class OSM_CLIP_Dataset(Dataset):
    def __init__(self, base_folder_path, preprocess, patch_size, image_size, json_path, mask_path):
        self.base_folder_path = (
            base_folder_path  # Base folder containing the images and json files
        )
        self.patch_size = patch_size
        self.image_size = image_size
        self.preprocess = preprocess
        self.json_path = json_path
        self.mask_path = mask_path

        # self.data = os.listdir(os.path.join(self.base_folder_path, json_path))

        # We always want to make sure the dictionaries are not empty
        self.data = []
        folder_path = os.path.join(self.base_folder_path, self.json_path)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data:  # Check if the dictionary is not empty
                self.data.append(file_name)

        self.totensor = ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data_path = os.path.join(self.base_folder_path, "osm_metadata", self.data[idx])
        data_path = os.path.join(self.base_folder_path, self.json_path, self.data[idx])

        # Open the image with PIL
        image_path = os.path.join(
            self.base_folder_path, "images", self.data[idx].replace(".json", ".png")
        )
        image = Image.open(
            image_path
        )  # .convert("RGB") TODO: check if it is enough to convert the image to RGB later in the preprocess function
        if image is None:
            raise RuntimeError(f"Image {self.data[idx]} not found")

        # we have to check if the mask exists, if not we have to create it, this will speed up the process
        mask_path = os.path.join(
            self.base_folder_path,
            self.mask_path,
            self.data[idx].replace(".json", ".json"),
        )

        if not os.path.exists(mask_path):
            logging.info(
                f"Mask for {self.data[idx]} with patch size {self.patch_size} not found. Creating it."
            )
            masks = extract_masks_from_osm_image(
                osm_data_file=data_path,
                img_size=self.image_size,
                cell_size=self.patch_size,
            )

            with open(mask_path, "w") as f:
                json.dump(masks, f)
            # visualize_grid_masks(masks, self.image_size, self.patch_size, image_path)
        else:
            # Load the masks from the JSON file
            with open(mask_path, "r") as f:
                masks = json.load(f)

        image = self.preprocess(
            image
        )  # TODO: do we have to preprocess the image? here? we have no preprocessing and the image had size of 600x600
        return image, masks


def custom_collate_fn(batch, image_size, patch_size):
    tags_patches_batch = {}
    images = []

    grid_size = int(image_size / patch_size)

    for i, element in enumerate(batch):
        image, masks = element
        images.append(image)

        # Process the tags and patch numbers
        for tag, related_patches in masks.items():
            if tag != "{}":  # TODO: CHECK DATA!
                mask = tags_patches_batch.get(tag, [])
                for patch_number in related_patches:
                    offset_patch = patch_number + i * (
                        grid_size**2
                    )  # Map patches back to the original image
                    mask.append(offset_patch)
                tags_patches_batch[tag] = mask
            else:
                logging.critical(f"Empty tag found in image {i}")

    # REORDERING THE PATCHES ascendingly so that the smallest patch is always selected, tested in one example from 76 to 96 patches.
    tags_patches_batch = {
        k: v
        for k, v in sorted(tags_patches_batch.items(), key=lambda item: len(item[1]))
    }

    # In tags_patches_batch we have the tags as keys and the patch numbers as values. A patch can be assigned to more than one tag.
    patch_to_index = {}
    tags_patches_out = {}

    # Store all feasible patches for each tag instead of selecting one
    for tag, patches in tags_patches_batch.items():
        feasible_patches = [
            patch for patch in patches if patch not in patch_to_index
        ]  # Avoid duplicate assignments
        if feasible_patches:
            tags_patches_out[tag] = feasible_patches
            for patch in feasible_patches:
                patch_to_index[patch] = (
                    patch  # Keep patch identity instead of re-indexing
                )

    images = torch.stack(images)
    list_tags = list(tags_patches_out.keys())
    list_tags = prepare_tags(list_tags)
    tokenized_tags = clip.tokenize(list_tags, truncate=True)

    return images, tags_patches_out, tokenized_tags


def _pluck_patches(image_features, tags_patches_batch, image_size, patch_size):
    grid_size = int(image_size / patch_size)
    plucked_features = torch.zeros(
        len(tags_patches_batch),
        image_features.shape[2],
        device=image_features.device,
        dtype=image_features.dtype,
    )

    for i, (tag, patches) in enumerate(tags_patches_batch.items()):
        tag_feature_vectors = []

        for patch_number in patches:
            image_number = patch_number // (
                grid_size**2
            )  # Identify which image this patch belongs to
            patch_index = patch_number % (
                grid_size**2
            )  # Identify the patch index within the image
            tag_feature_vectors.append(image_features[image_number, patch_index, :])

        if len(tag_feature_vectors) > 0:
            plucked_features[i] = torch.stack(tag_feature_vectors).mean(
                dim=0
            )  # Store mean feature vector

    return plucked_features


class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        el = self.data[idx]
        filename = el["filename"]
        caption = el["caption"]
        dataset = el["dataset"]
        if dataset == "RSICD":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/RSICD_captions/images/{filename}.jpg"
            )
        elif dataset == "UCM":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/UCM_captions/images/{filename}.tif"
            )
        elif dataset == "SIDNEY":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/SIDNEY_Captions/images/{filename}.tif"
            )
        elif dataset == "NWPU":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/NWPU_Captions/nwpu/images/{el['category']}/{filename}.jpg"
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        image = self.preprocess(image)
        text_inputs = clip.tokenize(caption)
        return image, text_inputs


class AugmentedGeneralDataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess, aug_new_images=False, num_of_aug_images=0):
        self.data = data
        self.preprocess = preprocess
        self.aug_new_images = aug_new_images
        self.num_of_aug_images = num_of_aug_images

        # Define augmentation transforms
        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        )

    def __len__(self):
        if self.aug_new_images:
            return len(self.data) * (1 + self.num_of_aug_images)
        return len(self.data)

    def __getitem__(self, idx):
        # Determine if this is an original or augmented item
        is_augmented = False
        aug_version = 0

        if self.aug_new_images and self.num_of_aug_images > 0:
            # Calculate original data index and augmentation version
            original_idx = idx % len(self.data)
            aug_version = idx // len(self.data)
            is_augmented = aug_version > 0
        else:
            original_idx = idx

        el = self.data[original_idx]
        filename = el["filename"]
        caption = el["caption"]
        dataset = el["dataset"]

        # Load the image
        if dataset == "RSICD":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/RSICD_captions/images/{filename}.jpg"
            )
        elif dataset == "UCM":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/UCM_captions/images/{filename}.tif"
            )
        elif dataset == "SIDNEY":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/SIDNEY_Captions/images/{filename}.tif"
            )
        elif dataset == "NWPU":
            image = Image.open(
                f"/media/Melgani/Riccardo/Datasets/image_captioning/NWPU_Captions/nwpu/images/{el['category']}/{filename}.jpg"
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        # Apply augmentation if this is an augmented version
        if is_augmented:
            # Use a different random seed for each augmentation version of the same image
            torch.manual_seed(original_idx * 100 + aug_version)
            image = self.augmentations(image)

        # Apply standard preprocessing
        image = self.preprocess(image)
        text_inputs = clip.tokenize(caption)

        return image, text_inputs


# Define custom collate function to handle captions
def custom_collate_fn_RSICD(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, captions