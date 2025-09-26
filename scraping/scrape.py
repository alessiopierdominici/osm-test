'''
This file contains functions and utilities to scrape data from OSM.
'''
import random 
import json
import time
from math import radians, cos
import requests
import os
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from itertools import product
from preprocessing import filter_tags, keep_only_tagged, intersect_elements, proj_lat_lon_on_image, remove_names, connect_elements, filter_outside_elements
from config import MAPBOX_ACCESS_TOKEN

def get_rbg_image(bbox:tuple, image_size:int=512):
    '''
    This function returns a RGB image from mapbox satellite-v9 style. 
    Input:
        bbox: (bottom, left, top, right) tuple of coordinates in decimal degrees of the bounding box
        image_size: the size of the image in pixels (supposing a square image)
    Output:
        image: RGB image
    '''
    
    south, west, north, east = bbox
    
    # Create the URL
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/[{west},{south},{east},{north}]/{image_size}x{image_size}?access_token={MAPBOX_ACCESS_TOKEN}"

    # Fetch the image
    try:
        response = requests.get(url)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        else:
            print(f"Failed to get image: {response.content}")
            return None
    except:
        pass

def fetch_overpass_data(bbox):
    """
    Fetch OSM data within a bounding box using the Overpass API.
    
    Parameters:
    bbox: (bottom, left, top, right) tuple of coordinates in decimal degrees of the bounding box
    
    Returns:
    str: OSM data as a string in JSON format.
    """
    
    
    south, west, north, east = bbox
    
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_url = "https://overpass.private.coffee/api/interpreter"
    overpass_url = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    
    # Define the Overpass QL query
    overpass_query = f"""
    [out:json];
    (
      node({south},{west},{north},{east});
      way({south},{west},{north},{east});
    );
    out geom;
    """
    # relation({bottom},{left},{top},{right});
    # Make the API request
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return f"Failed to fetch data: {response.status_code}"


def random_location_with_bounding_box(box_height:int=200, box_width:int=200):
    '''
    Generates a random location on the globe, and return a bounding box of max size 200x200 meters around this random location. 
    
    Input:
    box_height: the height of the bounding box in meters
    box_width: the width of the bounding box in meters
    
    Output:
    bbox: the coordinates of the bounding box (south, west, north, east)
    '''
    # Generate a random latitude and longitude
    lat = random.uniform(25, 49)
    lon = random.uniform(-125, -66)
    
    # Convert meters to degrees
    meters_per_degree_lat = 111320  # One degree of latitude in meters
    meters_per_degree_lon = meters_per_degree_lat * cos(radians(lat))  # Adjust for longitude

    # Calculate the half-height and half-width in degrees
    delta_lat = box_height / 2 / meters_per_degree_lat
    delta_lon = box_width / 2 / meters_per_degree_lon

    # Calculate the bounding box
    north = lat + delta_lat
    south = lat - delta_lat
    east = lon + delta_lon
    west = lon - delta_lon
    
    bbox = (south, west, north, east)
    return bbox

def scrape_data_from_OSM(n_samples:int, box_width:int, box_height:int, res_image:int, saving_folder:str="scraped_data", taginfo="taginfo-wiki.csv", divide_in_four:bool=False):
    '''
    Scrapes a total of n_samples of couples image-OSMdata. 
    It only keeps location in which there is at least one OSM entity (either a node or a way).
    Saves the OSM data in the folder saving_folder, and the images in saving_folder/images. 
    OSM data is saved as a json file, with bounding box coordinates (south, west, north, east) as the name of the file.
    
    Input:
    n_data_points: number of data samples to obtain
    box_width: the width of the bounding box in meters
    box_height: the height of the bounding box in meters
    res_image: the resolution of the image in pixels (always a square image)
    saving_folder: the folder in which to save the results
    
    Output: 
    None
    
    '''
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
        
    if not os.path.exists(os.path.join(saving_folder, "images")):
        os.makedirs(os.path.join(saving_folder, "images"))
    
    if not os.path.exists(os.path.join(saving_folder, "raw_osm_data")):
        os.makedirs(os.path.join(saving_folder, "raw_osm_data"))
    
    if divide_in_four:
        if not os.path.exists(os.path.join(saving_folder, "original_tiles")):
            os.makedirs(os.path.join(saving_folder, "original_tiles"))
        
    data = 0 
    
    taginfo = pd.read_csv(taginfo)
    
    # Initialize the progress bar with a total of n_samples
    pbar = tqdm(total=n_samples, desc="Data Progress")
    
    if divide_in_four:
        if 1280-res_image*2<50: 
            raise Warning("The difference between the new resolution and the old one is too small, artifacts can be produced.")
        new_res_image = 1280
        box_height = box_height*2
        box_width = box_width*2
        
    while True:
        try:
            if data==n_samples:
                break
            
            # Extract random bbox 
            bbox = random_location_with_bounding_box(box_height=box_height, box_width=box_width)
            #try:
            osm_data = fetch_overpass_data(bbox)
            if type(osm_data)!=str:
                # Check if there is at least one element that has tags.
                elements = keep_only_tagged(osm_data["elements"])
                #print("After tag filtering: ", len(elements))
                elements = filter_tags(elements, taginfo)
                #print("After filtering tags: ", len(elements))
                elements = remove_names(elements, taginfo)
                #print("After removing names: ", len(elements))
                elements = connect_elements(elements)
                #print("After connection: ", len(elements))
                elements = intersect_elements(elements, bbox)
                #print("After intersection: ", len(elements))
                elements = filter_outside_elements(elements, bbox)
                #print("After filtering outside: ", len(elements))
                elements = proj_lat_lon_on_image(elements, bbox)
                #print("After projection: ", len(elements))
                if len(elements)>=3:
                    data+=1
                    pbar.update(1)
                    if not divide_in_four:
                        # Directly save OSM data
                        image = get_rbg_image(bbox, image_size=res_image)
                        with open(saving_folder+"/osm_data/"+str(bbox[0])+"_"+str(bbox[1])+"_"+str(bbox[2])+"_"+str(bbox[3])+".json", "w") as f:
                            json.dump(osm_data, f, indent=4)
                        image.save(saving_folder+"/images/"+str(bbox[0])+"_"+str(bbox[1])+"_"+str(bbox[2])+"_"+str(bbox[3])+".png")
                    else:
                        image = get_rbg_image(bbox, image_size=new_res_image)
                        image.save(saving_folder+"/original_tiles/"+str(bbox[0])+"_"+str(bbox[1])+"_"+str(bbox[2])+"_"+str(bbox[3])+".png")

                time.sleep(0.5)
        except KeyboardInterrupt:
            break
        except:
            continue
    
    pbar.close()
    
    print("Downloaded a total of "+str(data)+" samples.")
    
    if divide_in_four:
        meters_per_degree_lat = 111320
        pbar2 = tqdm(total=len(os.listdir(saving_folder+"/original_tiles")), desc="Division of big tiles into 4x4 patches")
        # Process OSM DATA
        for image_filename in os.listdir(saving_folder+"/original_tiles"):
            original_bbox = tuple([float(x) for x in image_filename.split(".png")[0].split("_")])
            image = Image.open(saving_folder+"/original_tiles/"+image_filename)
            # Crop the image
            new_dim = res_image*2
            # Remove the bottom row of pixels
            image = image.crop((0,0,new_dim,new_dim))
            # Modify the original bbox 
            center_lat = (original_bbox[0]+original_bbox[2])/2
            meters_per_degree_lon = meters_per_degree_lat*cos(radians(center_lat))
            meters_to_cut = box_height*(1280-new_dim)/1280
            original_bbox = original_bbox[0]+(meters_to_cut/meters_per_degree_lat), original_bbox[1], original_bbox[2], original_bbox[3]-(meters_to_cut/meters_per_degree_lon)
            # Divide in four the image and the bbox
            lon_diff = (original_bbox[3]-original_bbox[1])/2 # east - west
            lat_diff = (original_bbox[2]-original_bbox[0])/2 # north - south
            # Create sub bounding boxes for the four quadrants
            sub_bbox3 = (original_bbox[0], original_bbox[1], original_bbox[0]+lat_diff, original_bbox[1]+lon_diff)
            
            sub_bbox4 = (original_bbox[0], original_bbox[1]+lon_diff, original_bbox[0]+lat_diff, original_bbox[3])
            
            sub_bbox1 = (original_bbox[0]+lat_diff, original_bbox[1], original_bbox[2], original_bbox[1]+lon_diff)
            
            sub_bbox2 = (original_bbox[0]+lat_diff, original_bbox[1]+lon_diff, original_bbox[2], original_bbox[3])
            
            subboxes = [sub_bbox1, sub_bbox2, sub_bbox3, sub_bbox4]
            
            h = new_dim
            w = new_dim
            d = res_image
            
            grid = product(range(0, h-h%d, d), range(0, w-w%d, d))

            tile = 0 
            for i, j in grid:
                box = (j, i, j+d, i+d)
                subbox = subboxes[tile]
                # Get osm data
                # New filename using the subbox coordinates
                filename = str(subbox[0])+"_"+str(subbox[1])+"_"+str(subbox[2])+"_"+str(subbox[3])+".png"
                if filename in os.listdir(saving_folder+"/images"):
                    continue
                else:
                    try:
                        osm_data = fetch_overpass_data(subbox)
                        image_out_path = os.path.join(saving_folder+"/images/", filename)
                        image.crop(box).save(image_out_path)
                        with open(os.path.join(saving_folder+"/raw_osm_data/", filename.replace(".png",".json")), "w") as f:
                            json.dump(osm_data, f, indent=4)
                        tile+=1
                        time.sleep(0.5)
                    except:
                        print("poie")
            
            pbar2.update(1)
        pbar2.close()
        
        print("Finished division, total number of images generated: "+str(len(os.listdir(saving_folder+"/original_tiles"))*4))
    
if __name__=="__main__":
    # ee.Authenticate()
    # ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    # HYPERPARAMETERS FOR TILE SCRAPING #
    area_side = 250 # side in METERS of the square area that will be exported
    image_resolution = 600 # resolution of the single patch!
    saving_folder = "/media/data_fast/Riccardo/new_osm_data/"
    divide_in_four=True # If true, it will download a bigger area and divide it in four smaller patches. To download more data with the free tier of Mapbox
    n_images = 49000 # 50000/month free tier from Mapbox
    #####################################
    
    scrape_data_from_OSM(n_images, area_side, area_side, image_resolution, saving_folder=saving_folder, divide_in_four=divide_in_four)