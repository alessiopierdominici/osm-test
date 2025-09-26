'''
This functions basically converts raw osm data retrieved from the Overpass API into data that can be used to build the dataset for visual question answering.
These functions does not alter the raw data, just process it to create the visual question answering dataset. 

Preprocessing:

starting with raw data, open with json and select the elements data["elements"] (the other is info about the osm API, that is not useful):

1. remove_pure_nodes: gets a list of elements. It separates the tagged elements from the untagged elements (pure nodes usually). 
   returns: filtered_elements, pure_nodes

2. filter_tags: gets a list of filtered elements, a list of all accepted keys from OSM and a pandas dataframe with the taginfo. It filters out tags that 
                based on the list of keys and the taginfo. The goal is to have a list of elements with only the tags that are most useful for the VQA task.
                If an element does not contain any useful tag, it is removed completely. 
   returns: new_elements

3. intersect_and_project: gets a list of elements with tags already filtered. It computes the intersection of the elements with the image bounding box.
                          For closed ways, it computes the interesection between two polygons. 
                          For open ways, it proceed chunk by chunk, evaluating the intersection of each segment with the image bounding box.
'''

import math
import pandas as pd
from utils.general_utils import check_closed_way, get_center_way, lat_lon_to_pixel_coords, check_point_inside_bbox, is_represented
from shapely.geometry import Polygon, LineString, Point
from tqdm import tqdm
import json
import os

def filter_tags(elements:list, taginfo:pd.DataFrame)->list:
    '''
    This function examine each element tags and remove the one that are not in "taginfo-wiki.csv", because it means that they are sneaky. 
    If an element remains without tags, it is removed.
    
    Input: 
        elements: a list of elements (nodes or ways)
        taginfo: pandas dataframe containing all the informations about tags. Used to filter the tags. 
    Output:
        new_elements: the list of elements with filtered tags.
    '''
    all_keys = list(taginfo["key"])
    new_elements = []
    for element in elements:
        tags = element["tags"]
        filtered_tags = {}
        if "layer" in tags.keys() and tags["layer"]<"0":
            # Feature not visible
            continue
        if "location" in tags.keys() and (tags["location"]=="underground" or tags["location"]=="underwater"):
            # Feature not visible
            continue
        for key in tags.keys():
            if "tiger:" in key:
                if key=="tiger:county":
                    filtered_tags["county"] = tags[key]
                else:
                    continue
            elif key =="ele" or key=="noexit" or key=="fixme" or key=="height" or "height" in key or key=="maxheight":
                continue
            elif key in all_keys:
                # Get the tgroup 
                tgroup = taginfo[taginfo["key"]==key]["tgroup"].values[0]
                if tgroup!="references" and tgroup!="import" and tgroup!="annotations" and tgroup!="addresses" and tgroup!="boundaries":
                    filtered_tags[key] = tags[key]
            else:
                continue
        
        if len(filtered_tags) != 0:
            if "name" in filtered_tags.keys() and "noname" in filtered_tags.keys() and "yes" in filtered_tags["noname"]:
                filtered_tags.pop("noname")
            if "name" in filtered_tags.keys() and "alt_name" in filtered_tags.keys():
                filtered_tags.pop("alt_name")
                
            filtered_tags["position"] = tags["position"]
            element["tags"] = filtered_tags
            new_elements.append(element)
            
    return new_elements

def find_intersection(image_bbox:list, element:dict) -> dict:
    '''
    Finds the intersections between the image bbox and the elements. 
    
    Input: 
    image bbox: the bounding box delimitating the image (south, west, north, east).
    elements: a osm element to intersect with the image bbox.
    enclosing_bbox: the smaller bbox enclosing all the elements (for normalization purposes).
    
    Output:
    intersected_elements: a new list of all elements with coordinates at most inside the image bbox
    '''

    i_bbox = Polygon([(image_bbox[0], image_bbox[1]), (image_bbox[0], image_bbox[3]), (image_bbox[2], image_bbox[3]), (image_bbox[2], image_bbox[1]), (image_bbox[0], image_bbox[1])])
    geometry = element["geometry"]
    
    if check_closed_way(element):
        # In this case it is an area-area intersection
        element_poly = Polygon([(node["lat"], node["lon"]) for node in geometry])
        intersection_shapely = element_poly.intersection(i_bbox)
        if not (intersection_shapely.is_empty):
            # See if it is a multipolygon
            if intersection_shapely.geom_type == "MultiPolygon":
                polygons = list(intersection_shapely.geoms)
                intersections = []
                for polygon in polygons:
                    temp_intersection = []
                    for point in list(polygon.exterior.coords):
                        temp_intersection.append({'lat': point[0],'lon': point[1]})
                        # If it was closed, it should be closed also now
                        temp_intersection.append({'lat': temp_intersection[0]["lat"], 'lon': temp_intersection[0]["lon"]})
                    intersections.append(temp_intersection)
                intersection = intersections
            else:
                intersection = [{'lat': point[0],'lon': point[1]} for point in list(intersection_shapely.exterior.coords)]
                # If it was closed, it should be closed also now
                intersection.append({'lat': intersection[0]["lat"], 'lon': intersection[0]["lon"]})
    else:
        # Line - area intersection
        intersection = []
        if check_point_inside_bbox(geometry[0], i_bbox):
            intersection.append({'lat': geometry[0]["lat"], 'lon': geometry[0]["lon"]})
            
        for i in range(0, len(geometry)-1): # considering segment by segment
            line = LineString([(geometry[i]["lat"], geometry[i]["lon"]), (geometry[i+1]["lat"], geometry[i+1]["lon"])])
            first_point = Point(geometry[i]["lat"], geometry[i]["lon"])
            last_point = Point(geometry[i+1]["lat"], geometry[i+1]["lon"])
            intersection_line = line.intersection(i_bbox)
            # Check the intersection is not empty
            if not (intersection_line.is_empty):
                first = Point(intersection_line.coords[0])
                last = Point(intersection_line.coords[-1])
                if (first!=first_point) and (last==last_point):
                    # Incoming line
                    intersection.append({'lat': intersection_line.coords[0][0], 'lon': intersection_line.coords[0][1]})
                elif (first==first_point) and (last!=last_point):
                    intersection.append({'lat': intersection_line.coords[0][0], 'lon': intersection_line.coords[0][1]})
                    intersection.append({'lat': intersection_line.coords[-1][0], 'lon': intersection_line.coords[-1][1]})
                elif (first!=first_point) and (last!=last_point):
                    # Line crossing the box 
                    intersection.append({'lat': intersection_line.coords[0][0], 'lon': intersection_line.coords[0][1]})
                    intersection.append({'lat': intersection_line.coords[-1][0], 'lon': intersection_line.coords[-1][1]})
                elif (first==first_point) and (last==last_point):
                    # Line completely inside the box
                    intersection.append({'lat': geometry[i]["lat"], 'lon': geometry[i]["lon"]})
            
        if check_point_inside_bbox(geometry[-1], i_bbox):
            intersection.append({'lat': geometry[-1]["lat"], 'lon': geometry[-1]["lon"]})
    
    # Sanity check on intersection
    for point in intersection:
        if type(point)==list:
            # Multipolygon
            for p in point:
                if p["lat"]<image_bbox[0] or p["lat"]>image_bbox[2] or p["lon"]<image_bbox[1] or p["lon"]>image_bbox[3]:
                    raise ValueError("Intersection out of enclosing bbox!")
        else:
            # Single polygon intersection, or line intesection
            if point["lat"]<image_bbox[0] or point["lat"]>image_bbox[2] or point["lon"]<image_bbox[1] or point["lon"]>image_bbox[3]:
                raise ValueError("Intersection out of enclosing bbox!")
            
    element["intersection"] = intersection
    return element
    
def proj_lat_lon_on_image(elements:list[dict], image_bbox:list[float]):
    '''
    This function takes a list of osm elements and project the location based on the given bounding box. 
    It divides the image in 9 portions, as\n
    |-----------------|-----------------|-----------------|\n
    |-----------------|-----------------|-----------------|\n
    |----Top Left-----|---Top Center----|----Top Right----|\n
    |-----------------|-----------------|-----------------|\n
    |-----------------|-----------------|-----------------|\n
    |-----------------|-----------------|-----------------|\n
    |   Center Left   |   Center        |   Center Right  |\n
    |-----------------|-----------------|-----------------|\n
    |-----------------|-----------------|-----------------|\n
    |-----------------|-----------------|-----------------|\n
    |   Bottom Left   |  Bottom Center  |   Bottom Right  |\n
    |-----------------|-----------------|-----------------|\n
    |-----------------|-----------------|-----------------|\n
    
    Input: 
        bbox: the bounding box delimitating the image (west, south, east, north).
        elements: the list of osm elements (all must be tagged).
        
    Output:
        located_elements: the list of elements with the "position" key, that indicates the broad position of the element in the image.
    '''
    south, west, north, east = image_bbox
    # Calculate the increments for latitude and longitude to split into three equal parts.
    lat_increment = (north - south) / 3
    lon_increment = (east - west) / 3

    # Calculate the starting points for latitude and longitude (bottom left).
    start_lat = south + lat_increment / 2
    start_lon = west + lon_increment / 2

    # List to hold the center of each tile
    tiles_centers = [
        (start_lat + i * lat_increment, start_lon + j * lon_increment)
        for i in range(3) for j in range(3)
    ]

    placing_identifiers = ["bottom left", "bottom center", "bottom right", "center left", "center", "center right", "top left", "top center", "top right"]
    
    # Convert the element position in (lat,lon) to a textual identifier, based on the closest tile center.
    # Approximate using the euclidean distance on latitude and longitude coordinates.
    
    located_elements = []
    for element in elements:
        assert "tags" in element.keys(), "Error in the prior filtering of nodes!"
        if element["type"]=="node":
            lat, lon = element["lat"], element["lon"]
            # Get the closest center 
            min_dist = float("inf")
            index = 0
            for i, center in enumerate(tiles_centers):
                dist = math.sqrt((lat - center[0])**2 + (lon - center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    index = i
            
            element["tags"]["position"] = placing_identifiers[index]
            
        elif element["type"]=="way":
            if check_closed_way(element):
                lat, lon = get_center_way(element)
                element["tags"]["position"] = ""
                already_placed = []
                for j in range(len(lat)):
                    if j!=0:
                        element["tags"]["position"] += " and "
                    # Get the closest center 
                    min_dist = float("inf")
                    index = 0
                    for i, center in enumerate(tiles_centers):
                        dist = math.sqrt((lat[j] - center[0])**2 + (lon[j] - center[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            index = i
                    if index not in already_placed:        
                        element["tags"]["position"] += placing_identifiers[index]
                        already_placed.append(index)
                    else:
                        element["tags"]["position"] = element["tags"]["position"][:-5]
                    
            else:
                # Handle the open ways (roads-rivers)
                nodes_inside = element["intersection"]
                # Get the closest center for the starting point
                min_dist = float("inf")
                indexstart = 0
                for i, center in enumerate(tiles_centers):
                    dist = math.sqrt((nodes_inside[0]["lat"] - center[0])**2 + (nodes_inside[0]["lon"] - center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        indexstart = i
                
                # Get the closest center for the ending point
                min_dist = float("inf")
                indexend = 0
                for i, center in enumerate(tiles_centers):
                    dist = math.sqrt((nodes_inside[-1]["lat"] - center[0])**2 + (nodes_inside[-1]["lon"] - center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        indexend = i
                
                if indexstart == indexend:
                    element["tags"]["position"] = placing_identifiers[indexstart]
                else:
                    element["tags"]["position"] = "from " + placing_identifiers[indexstart] + " to " + placing_identifiers[indexend]
                
        else:
            raise ValueError("Element type not yet supported!")
        
        located_elements.append(element)
    
    return located_elements

def intersect_elements(filtered_elements:list[dict], image_bbox:list[float]) -> list[dict]:
    '''
    This function finds the intersections between the elements. Only ways are considered, as nodes are not intersecting.
    
    Input: 
        elements: the list of osm elements.
        image_bbox: the bounding box delimitating the image (south, west, north, east).
        pure_nodes: the list of nodes that are inside the image bbox as returned by OSM api.
    
    Output:
        intersected_elements: a new list of all elements with a new key (intersection) representing the coordinates of intersection of all the ways with the image bbox.
    '''
    intersected_elements = []
    for element in filtered_elements:
        if element["type"]=="node":
            intersected_elements.append(element)
        elif element["type"]=="way":
            intersected = find_intersection(image_bbox, element)
            intersected_elements.append(intersected)
        
    return intersected_elements

def eliminate_duplicate_elements(elements:list[dict]) ->list[dict]:
    '''
    This function eliminates duplicate elements from the list. 
    Input:
        elements: the list of osm elements.
        
    Output:
        filtered_elements: the list of osm elements without duplicates (it removes entries that are identical).
    '''
    already_seen_tags = []
    filtered_elements = []
    for element in elements:
        if element["tags"] not in already_seen_tags:
            already_seen_tags.append(element["tags"])
            filtered_elements.append(element)
    
    further_filtered_elements = []
    for i in range(len(filtered_elements)):
        to_add = True
        for j in range(len(filtered_elements)):
            if i!=j:
                if is_represented(filtered_elements[i]["tags"], filtered_elements[j]["tags"]):
                    to_add=False
        if to_add:    
            further_filtered_elements.append(filtered_elements[i])
        
    return further_filtered_elements

def remove_names(elements:list[dict], taginfo:pd.DataFrame, all_category:bool=True) -> list[dict]:
    '''
    This function removes the "name" and "alt_name" entries from the elements tags. 
    It is used to avoid the model to learn the names of the elements, as they are too specific information for the model.
    
    Input:
        elements: the list of osm elements.
    Output:
        filtered_elements: the list of osm elements without names.
    '''
    filtered_elements = []
    for element in elements:
        if "layer" in element["tags"].keys() and element["tags"]["layer"]<"0":
            continue
        if "location" in element["tags"].keys() and (element["tags"]["location"]=="underground" or element["tags"]["location"]=="underwater"):
            continue
        if all_category:
            new_tags = {}
            for key, value in element["tags"].items():
                if "name" in key or "county" in key or "operator" in key:
                    continue
                if "county" in value:
                    continue
                if key!="position":
                    tgroup = taginfo[taginfo["key"]==key]["tgroup"].values[0]
                    if tgroup!="names":
                        new_tags[key] = element["tags"][key]
                else:
                    new_tags[key] = element["tags"][key]
            
            element["tags"] = new_tags
        else:
            if "name" in element["tags"].keys():
                element["tags"].pop("name")
            if "noname" in element["tags"].keys():
                element["tags"].pop("noname")
            if "alt_name" in element["tags"].keys():
                element["tags"].pop("alt_name")
        
        filtered_elements.append(element)
    
    return filtered_elements

def remove_names_element(element:dict, taginfo:pd.DataFrame, all_category:bool=True) -> list[dict]:
    '''
    This function removes the "name" and "alt_name" entries from the elements tags. 
    It is used to avoid the model to learn the names of the elements, as they are too specific information for the model.
    
    Input:
        elements: the list of osm elements.
    Output:
        filtered_elements: the list of osm elements without names.
    '''
    if "layer" in element["tags"].keys() and element["tags"]["layer"]<"0":
        element=None
    if "location" in element["tags"].keys() and (element["tags"]["location"]=="underground" or element["tags"]["location"]=="underwater"):
        element=None
    
    if all_category:
        new_tags = {}
        for key, value in element["tags"].items():
            if "name" in key or "county" in key or "operator" in key:
                continue
            if "county" in value:
                continue
            if key!="position":
                tgroup = taginfo[taginfo["key"]==key]["tgroup"].values[0]
                if tgroup!="names":
                    new_tags[key] = element["tags"][key]
            else:
                continue
                #new_tags[key] = element["tags"][key]
        
        element["tags"] = new_tags
    else:
        if "name" in element["tags"].keys():
            element["tags"].pop("name")
        if "noname" in element["tags"].keys():
            element["tags"].pop("noname")
        if "alt_name" in element["tags"].keys():
            element["tags"].pop("alt_name")
            
    return element

def return_shape(element:dict)->list[list[float]]:
    '''
    This function returns the intersection of the element with the image. If no intersection nor geometry is found (should not happen) returns None.
    '''
    if element["type"]=="node":
        shape = [[element["lat"], element["lon"]]]
        
    elif element["type"]=="way":
        shape = []
        boundary = element["intersection"]
        for point in boundary:
            if type(point)==list:
                for p in point:
                    shape.append([p["lat"], p["lon"]])
                shape.append([]) # Append empty list
            else:
                shape.append([point["lat"], point["lon"]])
                
        if type(point)==list:
            shape = shape[:-1]
            
    return shape

def quantize_boundary(boundary:list[list[float]], image_resolution:int, image_bbox:list[float])-> list[list[int]]:
    '''
    This function quantize a boundary (in lat, lon) of the element to be integers projected in the image space.
    '''
    quantized_boundary = []
    for point in boundary:
        if len(point)==0:
            quantized_boundary.append([])
        else:
            pixel_coords = lat_lon_to_pixel_coords(image_bbox, point[0], point[1], (image_resolution,image_resolution))
            if(pixel_coords is not None):
                quantized_boundary.append([pixel_coords[0], pixel_coords[1]])
                    
    if len(quantized_boundary)==0:
        raise ValueError("Empty boundary!")
    
    return quantized_boundary

def convert_shape_tokens(boundary:list[list[int]]):
    '''
    This function converts a sequence of points in a sequence of tokens.
    '''
    converted_shape = []
    for point in boundary:
        if len(point)==0:
            converted_shape.append("multi")
        else:
            converted_shape.append(str(point[0])+" "+str(point[1]))

    return converted_shape

def transform_polygon_to_bbox(elements:list, im_bbox:list[float])-> list:
    '''
    This function takes a list of elements and transform the polygons to (rotated) bounding boxes. 
    
    Input:
    - elements: the list of osm elements
    
    Output:
    - elements: the list of osm elements with the polygons transformed to bounding boxes
    '''
    for element in elements:
        if element["type"]=="node":
            continue
        elif element["type"]=="way":
            image_bbox_poly = Polygon([[im_bbox[0], im_bbox[1]], [im_bbox[0], im_bbox[3]], [im_bbox[2], im_bbox[3]], [im_bbox[2], im_bbox[1]]])
            bbox = []
            if check_closed_way(element):
                boundary = element["geometry"]
                boundary_coords = []
                for point in boundary:
                    p_poly = Point(point["lat"], point["lon"])
                    if image_bbox_poly.contains(p_poly):
                        boundary_coords.append([point["lat"], point["lon"]])
                
                if boundary_coords[0]!=boundary_coords[-1]:
                    boundary_coords.append(boundary_coords[0])
    
                object_poly = Polygon(boundary_coords)
                if "tags" in element.keys():
                    print(element["tags"])
                # Get the rotated bbox 
                rot_boox = object_poly.minimum_rotated_rectangle
                for point in rot_boox.exterior.coords:
                    bbox.append({"lat":point[0], "lon":point[1]})
                element["bbox"] = bbox
            else:
                print(element["geometry"])
    return elements

def keep_only_tagged(elements:list[dict])->list[dict]:
    '''
    This function removes all the elements that are not tagged. 
    '''
    new_elements = []
    for element in elements:
        if "tags" in element.keys():
            new_elements.append(element)
    
    return new_elements

def post_process_osm_data(base_folder_dataset:str):
    taginfo = "taginfo-wiki.csv"
    taginfo = pd.read_csv(taginfo)
    
    # Initialize the progress bar with a total of n_samples
    raw_data_path = os.path.join(base_folder_dataset, "raw_osm_data")
    if not os.path.exists(os.path.join(base_folder_dataset, "postprocessed_osm_data")):
        os.makedirs(os.path.join(base_folder_dataset, "postprocessed_osm_data"))
        
    postprocessed_data_path = os.path.join(base_folder_dataset, "postprocessed_osm_data")
    
    pbar = tqdm(total=len(os.listdir(raw_data_path)), desc="Data Progress")
    tot_data = 0
    for file in os.listdir(raw_data_path):
        if file.endswith(".json") and not os.path.exists(os.path.join(postprocessed_data_path, file)):
            raw_data = json.load(open(os.path.join(raw_data_path, file)))
            image_bbox = [float(x) for x in file[:-5].split("_")[:4]]
            # Intersect everything first
            elements = keep_only_tagged(raw_data["elements"])
            elements = intersect_elements(elements, image_bbox)
            elements = proj_lat_lon_on_image(elements, image_bbox)
            elements = filter_tags(elements, taginfo)
            elements = eliminate_duplicate_elements(elements)
            if len(elements) != 0:
                # Save the postprocessed data
                with open(os.path.join(postprocessed_data_path, file), "w") as f:
                    json.dump(elements, f)
                tot_data += 1
        pbar.update(1)
    
    print("Finished postprocessing OSM data. Total samples remaining: ", tot_data)

if __name__=="__main__":
    post_process_osm_data("examples_scraped")