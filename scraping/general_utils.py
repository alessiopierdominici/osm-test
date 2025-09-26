import random
from shapely import affinity
from shapely.geometry import Polygon, Point
from PIL import ImageTransform

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
    
def is_represented(element1:dict, element2:dict):
    '''
    Return True if element1 is fully represented by element2. 
    '''
    is_represented = True
    for key in element1.keys():
        if key not in element2.keys():
            is_represented = False
            break
        else:
            if element1[key] != element2[key]:
                is_represented = False
                break
            
    return is_represented
    

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
    
def generate_random_bbox(image_size:tuple[int,int], min_bbox_size:int=100):
    '''
    Generate a random bbox given the image size.
    '''
    bbox_width = random.randint(min_bbox_size, image_size[0])
    bbox_height = random.randint(min_bbox_size, image_size[1])
    center_x = random.randint(bbox_width//2, image_size[0] - bbox_width//2)
    center_y = random.randint(bbox_height//2, image_size[1] - bbox_height//2)
    x1 = center_x - bbox_width//2
    y1 = center_y - bbox_height//2
    x2 = center_x + bbox_width//2
    y2 = center_y + bbox_height//2
    random_rotation = random.randint(0, 360)
    bbox_poly = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1),(x1, y1)])
    rotated_bbox = affinity.rotate(bbox_poly, random_rotation)
    return rotated_bbox, (bbox_width, bbox_height)

def crop_image_bbox(image, bbox, dims):
    '''
    Crop the image using the bbox.
    '''
    transform = []
    for coord in bbox.exterior.coords:
        transform.append(coord[0])
        transform.append(coord[1])
    result = image.transform((dims[0],dims[1]), ImageTransform.QuadTransform(transform))
    return result

def evaluate_elements_inside_bbox(bbox_poly:Polygon, elements:list[dict]):
    '''
    Evaluate the elements inside the bbox.
    '''
    
    elements_inside = []
    for element in elements:
        if "tags" in element:
            # If the center is inside, the element is inside
            if element["type"] == "node":
                if check_point_inside_bbox({"lat":element["lat"], "lon":element["lon"]}, bbox_poly):
                    elements_inside.append(element)
            elif element["type"] == "way":
                if check_closed_way(element):
                    center = get_center_way(element)
                    if check_point_inside_bbox({"lat":center[0], "lon":center[1]}, bbox_poly):
                        elements_inside.append(element)
                else:
                    # Evaluate the intersection between the bbox and the way
                    intersection = bbox_poly.intersection(Polygon([(node["lat"], node["lon"]) for node in element["nodes"]]))
                    if not (intersection.is_empty):
                        elements_inside.append(element)
        
    return bbox_poly, elements_inside
    
def get_bbox_for_question(elements:list[dict]):
    while True:
        bbox, dims = generate_random_bbox((512,512))
        bbox, elements_inside = evaluate_elements_inside_bbox(bbox, elements)
        if len(elements_inside) > 1:
            return bbox, elements_inside

def partial_geom(geometry):
    '''
    Return a partial geometry given a geometry
    '''
    partial_geom = []
    for point in geometry:
        if random.random() > 0.5:
            partial_geom.append(point)
    
    return partial_geom
        
if __name__=="__main__":
    # Open a random image 
    from PIL import Image
    
    image = Image.open("examples_scraped/images/26.5930031321351_-108.50214139350447_26.595302808743075_-108.49956962343248.png")
    
    # Generate a random bbox
    bbox, dims = generate_random_bbox(image.size)
    
    cropped = crop_image_bbox(image, bbox, dims)
    
    cropped.save("cropped.png")
    
    
    