import json
import os 
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tqdm import tqdm

from utils.preprocessing import lat_lon_to_pixel_coords, \
                                check_closed_way

def visualize_elements_on_image(image:Image, bbox:list, elements:list, visualize="all", file_to_save=None):
    '''
    This function project the elements on the image and visualize their position. 
    
    Input: 
    image: the image to use for visualization
    bbox: the bounding box delimitating the image (west, south, east, north)
    elements: the list of osm elements 
    visualize: the type of elements to visualize. It can be "all", "nodes", "ways"
    
    '''
    im2 = image.copy()
    draw = ImageDraw.Draw(im2)
    
    elements_to_visualize = []
    for element in elements:
        if visualize == "all":
            elements_to_visualize.append(element)
        elif visualize == "nodes":
            if(element["type"] == "node"):
                elements_to_visualize.append(element)
        elif visualize == "ways":
            if(element["type"] == "way"):
                elements_to_visualize.append(element)
    if len(elements_to_visualize)==0:
        return 
    
    for element in elements_to_visualize:
        # If it is a node
        if element["type"] == "node":
            lat = element["lat"]
            lon = element["lon"]
            # Go from lat and lon to pixel coords
            pixel_coords = lat_lon_to_pixel_coords(bbox, lat, lon, image.size)
            if(pixel_coords is not None):
                # Project on the image
                ellipse = [pixel_coords[0]-4, pixel_coords[1]-4, pixel_coords[0]+4, pixel_coords[1]+4]
                draw.ellipse(ellipse, fill="red")
        
        # If it is a way
        if element["type"] == "way":
            points = element["intersection"]
            xs = []
            ys = []
            if check_closed_way(element):
                for point in points:
                    if type(point) == list:
                        for p in point:
                            lat = p["lat"]
                            lon = p["lon"]
                            pixel_coords = lat_lon_to_pixel_coords(bbox, lat, lon, image.size)
                            if(pixel_coords is not None):
                                # Project on the image
                                xs.append(pixel_coords[0])
                                ys.append(pixel_coords[1])
                    else:
                        lat = point["lat"]
                        lon = point["lon"]
                        pixel_coords = lat_lon_to_pixel_coords(bbox, lat, lon, image.size)
                        if(pixel_coords is not None):
                            # Project on the image
                            xs.append(pixel_coords[0])
                            ys.append(pixel_coords[1])
                draw.polygon(list(zip(xs,ys)), outline="red")
            else:
                for point in points:
                    lat = point["lat"]
                    lon = point["lon"]
                    pixel_coords = lat_lon_to_pixel_coords(bbox, lat, lon, image.size)
                    if(pixel_coords is not None):
                        # Project on the image
                        xs.append(pixel_coords[0])
                        ys.append(pixel_coords[1])
                        draw.line(list(zip(xs,ys)), fill="red")
                        
    img3 = Image.blend(image, im2, alpha=0.5)
    
    if file_to_save is not None:
        img3.save("visualized_elements/"+file_to_save+".png")
    

def visualize_bulk(data_folder:str, visualize="all", files_to_do=None):
    '''
    This function visualize the elements on all the images in bulk. It is used for debugging purposes only. 
    
    Input:
    - data_folder: the folder containing the scraped data. This folder should contain the subfolder "images" containing the images.
    
    Output:
    None
    '''
    print("Visualizing the elements on the images..")
    if not os.path.exists("visualized_elements"):
        os.mkdir("visualized_elements")
        
    files = os.listdir(os.path.join(data_folder,"raw_osm_data"))
    for file in tqdm(files):
        if files_to_do is not None:
            if file in files_to_do:
                assert file.endswith(".json")
                filename = file[:-5]
                elements = json.load(open(os.path.join(data_folder,"postprocessed_osm_data",filename+".json"), "r"))
                bbox = [float(x) for x in file[:-5].split("_")[:4]]
                image = Image.open(os.path.join(data_folder,"images",filename+".png"))
                visualize_elements_on_image(image, bbox, elements, visualize=visualize,file_to_save=filename)
                plt.close()
        else:
            assert file.endswith(".json")
            filename = file[:-5]
            elements = json.load(open(os.path.join(data_folder,"postprocessed_osm_data",filename+".json"), "r"))
            bbox = [float(x) for x in file[:-5].split("_")[:4]]
            image = Image.open(os.path.join(data_folder,"images",filename+".png"))
            visualize_elements_on_image(image, bbox, elements, visualize=visualize,file_to_save=filename)
            plt.close()
        
if __name__=="__main__":
    toy_dataset = "../OpenStreetWhat/toy_dataset"
    files = os.listdir(toy_dataset)
    visualize_bulk("/media/data_fast/Riccardo/test_osm_scraping", visualize="all", files_to_do=files)