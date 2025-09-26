'''
In this module there will be utilities to load training/testing data seamlessly.
Each dataset will have a function that given a split will return a list of tuples (image_path, label).
'''

import pandas as pd
import json

def UCM_captions_loader(split):
    # Check that the split is correct
    valid_splits = ["train", "val", "test"]
    assert split in valid_splits, f"Split must be one of {valid_splits}"
    # Open the filenames file 
    with open("finetuning_datasets/UCM_captions/filenames/filenames_"+split+".txt", "r") as f:
        filenames = f.readlines()
    
    # Open the descriptions file 
    with open("finetuning_datasets/UCM_captions/filenames/descriptions_UCM.txt", "r") as f:
        descriptions = f.readlines()
    
    filenames = [x.strip().split(".tif")[0] for x in filenames]
    
    data = []
    for descriptions in descriptions:
        file = descriptions.strip().split(" ")[0]
        description = " ".join(descriptions.strip().split(" ")[1:])
        if file in filenames:
            image_path = "finetuning_datasets/UCM_captions/images/"+file+".tif"
            data.append((image_path, description))
        
    return data

def RSICD_captions_loader(split):
    # Check that the split is correct
    valid_splits = ["train", "val", "test"]
    assert split in valid_splits, f"Split must be one of {valid_splits}"
    # Open the filenames file 
    with open("finetuning_datasets/RSICD_captions/filenames/filenames_"+split+".txt", "r") as f:
        filenames = f.readlines()
    
    # Open the descriptions file 
    with open("finetuning_datasets/RSICD_captions/filenames/descriptions_RSICD.txt", "r") as f:
        descriptions = f.readlines()
    
    filenames = [x.strip().split(".jpg")[0] for x in filenames]
    
    data = []
    for descriptions in descriptions:
        file = descriptions.strip().split(" ")[0]
        description = " ".join(descriptions.strip().split(" ")[1:])
        if file in filenames:
            image_path = "finetuning_datasets/RSICD_captions/images/"+file+".tif"
            data.append((image_path, description))
        
    return data

def RSITMD_caption_loader(split):
    valid_splits = ["train", "val", "test"]
    assert split in valid_splits, f"Split must be one of {valid_splits}"
    
    # Open the annotations csv file 
    rsitmd_labels = pd.read_csv("finetuning_datasets/RSITMD/labels/RSITMD.csv")
    data = []
    # For every filename in the "filename" column, check if the split is equal to the one passed as argument, if so, put the tuple filename, sentence in the data list
    for _, row in rsitmd_labels.iterrows():
        if row["split"]==split:
            data.append(("finetuning_datasets/RSITMD/images/"+row["filename"], row["sentence"]))
    
    return data

def parseDataset(dataset_path: str, dataset: str, image_extension: str): 
    train_set = set()
    val_set = set()
    test_set = set()

    train = []
    val = []
    test = []


    with open(f"{dataset_path}/filenames/filenames_train.txt", "r") as f:
        for line in f:
            train_set.add(line.strip().removesuffix(image_extension))

    with open(f"{dataset_path}/filenames/filenames_val.txt", "r") as f:
        for line in f:
            val_set.add(line.strip().removesuffix(image_extension))

    with open(f"{dataset_path}/filenames/filenames_test.txt", "r") as f:
        for line in f:
            test_set.add(line.strip().removesuffix(image_extension))

    with open(f"{dataset_path}/filenames/descriptions_{dataset}.txt", "r") as f:
        for line in f:
            arr = line.strip().split(' ')
            filename = arr[0]
            caption = ' '.join(arr[1:])
            entry = {'filename': filename, 'caption': caption, 'dataset': dataset}
            if arr[0] in train_set:
                train.append(entry)
            if arr[0] in val_set:
                val.append(entry)
            if arr[0] in test_set:
                test.append(entry)    
    return train, val, test


def parseDatasetFromJSON(json_path: str, dataset: str, image_extension: str):
    train, val, test = [], [], []

    with open(json_path, "r") as f:
        data = json.load(f)  # Load the whole JSON file

    for category, images in data.items():  # Loop through top-level keys (e.g., "airplane")
        for img_info in images:  # Each image entry is a dictionary
            filename = img_info["filename"].replace(image_extension, "")
            split = img_info["split"]

            # Collect all captions
            captions = [img_info["raw"]]
            for i in range(1, 5):  # There are multiple caption variations
                key = f"raw_{i}"
                if key in img_info:
                    captions.append(img_info[key])

            # Store each caption as a separate tuple
            for caption in captions:
                entry = {'filename': filename, 'caption': caption, 'dataset': dataset, 'category': category}
                if split == "train":
                    train.append(entry)
                elif split == "val":
                    val.append(entry)
                elif split == "test":
                    test.append(entry)

    return train, val, test


if __name__=="__main__":
    data = RSITMD_caption_loader("val")
    print(data)