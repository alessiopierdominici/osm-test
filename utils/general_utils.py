import os
from shapely.geometry import Polygon, Point
from datetime import datetime

import torch
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import (
    RandomChoice,
    Lambda,
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomVerticalFlip,
    RandomResizedCrop,
    ColorJitter,
    RandomHorizontalFlip,
)
from torchvision.transforms import functional as TF

def check_point_inside_bbox(point:dict, bbox:Polygon) -> bool:
    '''
    This function checks if a point is inside a bounding box using Shapely.
    
    Input:
        point: a dictionary containing "lat" and "lon" coordinates of the point. 
        bbox: area in which the point could be contained or not. shapely poly.
    Output:
        True if point inside bbox, otherwise False.
    '''
    point = Point(point["lat"], point["lon"])
    return bbox.contains(point)

def lat_lon_to_pixel_coords(bbox, lat, lon, bbox_pixel_dim):
    '''
    This function converts lat and lon coordinates inside a bounding box to equivalent pixel coordinates. 
    
    Input: 
    bbox: the bounding box (west, south, east, north)
    lat: the latitude of the point
    lon: the longitude of the point
    bbox_pixel_dim: the dimension of the image in pixels
    '''
    # NB: you have to use PIL to open the image!

    south, west, north, east = bbox
    

    x_factor = bbox_pixel_dim[0]/(east - west)
    y_factor = bbox_pixel_dim[1]/(north - south)
    
    
    x = int((lon - west) * x_factor)
    y = int((north - lat) * y_factor)

    return (x, y)

def check_closed_way(way:dict) -> bool:
    '''
    Check if a way is closed or not. Return a boolean yes or no. 
    
    Input: way (dictionary representing a way)
    Output: True if the way is closed, False otherwise.
    '''
    if way['nodes'][0] == way['nodes'][-1]:
        return True
    else:
        return False

def get_center_way(way:dict) -> tuple:
    '''
    This function computes the center of a closed way. Actually, it computes the center of just the part inside the image (intersection).
    
    Input: 
        way (dictionary representing a way)
    Output: 
        the center of the way as a tuple (lat, lon)
    '''
    if check_closed_way(way):
        # Proceed with processing
        if "intersection" not in way.keys():
            raise ValueError("You have first to compute the intersections of the elements inside the image!")
        else:
            centers_lat = []
            centers_long = []
            if type(way["intersection"][0])==list:
                # Handle multipolygons
                for piece in way["intersection"]:
                    lats = []
                    longs = []
                    for node in piece:
                        lats.append(node["lat"])
                        longs.append(node["lon"])
                    # Append
                    centers_lat.append(sum(lats) / len(lats))
                    centers_long.append(sum(longs) / len(longs))
            else:
                lats = []
                longs = []
                for node in way["intersection"]:
                    lats.append(node["lat"])
                    longs.append(node["lon"])
            # Get the center
            centers_lat.append(sum(lats) / len(lats))
            centers_long.append(sum(longs) / len(longs))
            
            return (centers_lat, centers_long)
    else:
        raise ValueError("The way is not closed!")

def prepare_tags(unique_tags):
    # TODO think of better way to prepare tags
    processed_tags = []
    for tag in unique_tags:
        tag = tag.replace("{", "")
        tag = tag.replace("}", "")
        tag = tag.replace("'", "")
        tag = tag.replace("=", "")
        tag = tag.replace(":", "")
        tag = tag.replace(",", "")
        processed_tags.append(tag)
    
    return processed_tags

def convert_models_to_fp32(model):
    '''	
    This function converts the model to fp32.	
    '''
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 
        
    from datetime import datetime

def get_formatted_datetime():
    """
    Returns the current date and time formatted as a string.
    
    Returns:
        str: Formatted date and time string in the format 'YYYYMMDD_HHMMSS'.
    """
    # Retrieve the current date and time
    current_datetime = datetime.now()
    
    # Format the date and time according to the specified format string
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')
    
    return formatted_datetime

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# def get_preprocess_aug(n_px):
#     crop_size = int(224 * 0.8)

#     return Compose([
#         RandomCrop(crop_size),
#         ColorJitter(),
#         RandomHorizontalFlip(0.5),
#         RandomVerticalFlip(0.5),
#         RandomResizedCrop(crop_size, scale=(0.8, 1.2), ratio=(1.0, 1.0)),

#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

def get_remoteclip_transform(n_px):
    """
    Image transformation pipeline that closely follows the RemoteCLIP paper's augmentation strategy.
    
    Args:
        n_px (int): The target size for resizing images
        
    Returns:
        A composition of transformations matching RemoteCLIP's augmentation approach
    """
    return Compose([
        # Random crops for resizing as mentioned in the paper
        RandomResizedCrop(n_px, scale=(0.8, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),
        
        # Random horizontal flips as mentioned in the paper
        RandomHorizontalFlip(p=0.5),
        
        # Random rotations of 0°, 90°, 180°, and 270° specifically mentioned for rotation invariance
        RandomChoice([
            Lambda(lambda x: x),  # 0 degrees (no rotation)
            Lambda(lambda x: TF.rotate(x, 90)),
            Lambda(lambda x: TF.rotate(x, 180)),
            Lambda(lambda x: TF.rotate(x, 270))
        ]),
        
        # Standard processing pipeline
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def inference_with_model(model, image, text):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

def save_checkpoint(FINETUNE_DIR, CONF_NAME, model, epoch):
    if epoch==0:
        epoch = "0"
    checkpoint_path = os.path.join(FINETUNE_DIR, f"{CONF_NAME}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

def initialize_deterministic_mode(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
