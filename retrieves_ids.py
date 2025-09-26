import os
import json
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import ast
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import LineString as ShapelyLineString

OUTPUT_VIS_DIR = "temp/retrieve_ids_images/"
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True) 
# delete all the files in the directory
for file in os.listdir(OUTPUT_VIS_DIR):
    os.remove(os.path.join(OUTPUT_VIS_DIR, file))

def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key}: {value}')

def extract_masks_from_osm_image(osm_data_file, img_size, cell_size):
    '''
    Extracts grid cell indices for each OSM element in the given OSM data file.
    
    Input: 
    - osm_data_file: str. Path to the OSM data JSON file.
    - img_size: int. The size of the image in pixels (assuming square).
    - cell_size: int. The size of each grid cell in pixels.
    
    Output:
    - masks: dict. Keys are string representations of element tags, values are lists of grid cell indices.
    '''
    
    # Define helper function to convert row, col to linear index
    def index_from_xy(row, col, grid_size):
        return row * grid_size + col
    
    # Create output directory if it doesn't exist, this function can be called also in the pipeline
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True) 
        
    # Calculate grid dimensions
    num_grids = img_size // cell_size
    
    # Extract coordinate bounds from the filename
    file_name = os.path.basename(osm_data_file)
    file_name_without_extension = os.path.splitext(file_name)[0]
    
    # Parse coordinate bounds
    try:
        coords = file_name_without_extension.split('_')
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates in filename, got {len(coords)}")
            
        min_lat, min_lon, max_lat, max_lon = map(float, coords)
        
        # Verify coordinate ordering
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        
        if lat_span <= 0 or lon_span <= 0:
            raise ValueError(f"Invalid coordinate bounds: lat_span={lat_span}, lon_span={lon_span}")
    except Exception as e:
        raise ValueError(f"Failed to parse coordinates from filename: {str(e)}")
    
    # Initialize dictionary for masks
    masks = {}
    
    # Load OSM data
    try:
        with open(osm_data_file, 'r') as f:
            osm_elements = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load OSM data: {str(e)}")
        
    # TODO: chech why sometimes the osm_elements is a list and sometimes is a dictionary, in osm_data_rich is a dict in osm_data is a list
    if isinstance(osm_elements, list):
        osm_elements = {element["id"]: element for element in osm_elements}
  
    # Process each OSM element
    for element_id, element in osm_elements.items():
        # Extract element tags and create a key for the masks dictionary
        tags = element.get('tags', {}).copy()
        tags.pop("position", None)  # Remove position tag if present
        tag_key = str(tags)  # Use string representation of tags as key
        
        # Skip elements without tags
        if not tags:
            continue
            
        # Get or create mask for this tag combination
        mask = masks.get(tag_key, [])
        
        # Process element based on type
        if element['type'] == 'node':
            # Handle point element (node)
            try:
                lat = element['lat']
                lon = element['lon']
                
                # Convert geographic coordinates to image coordinates
                # Note: y-coordinate is flipped because image coordinates start at top-left
                norm_x = (lon - min_lon) / lon_span * img_size
                norm_y = (1 - (lat - min_lat) / lat_span) * img_size
                
                # Determine which grid cell contains this point
                grid_col = int(norm_x / cell_size)
                grid_row = int(norm_y / cell_size)
                
                # Ensure we're within bounds (just in case)
                if 0 <= grid_row < num_grids and 0 <= grid_col < num_grids:
                    cell_index = index_from_xy(grid_row, grid_col, num_grids)
                    if cell_index not in mask:
                        mask.append(cell_index)
            except KeyError as e:
                print(f"Warning: Node element {element_id} missing coordinate: {str(e)}")
                continue
                
        elif element['type'] == 'way':
            # Handle line/polygon element (way)
            try:
                way_nodes = element.get('intersection', [])
                
                # Check if way_nodes is properly formatted
                if not way_nodes:
                    continue
                    
                coords_to_check = []
                if isinstance(way_nodes[0], list) or (isinstance(way_nodes[0], dict) and not 'lat' in way_nodes[0]):
                    # Multiple segments
                    coords_to_check = way_nodes
                else:
                    # Single segment
                    coords_to_check = [way_nodes]
                
                # Process each segment
                for node_segment in coords_to_check:
                    # Extract latitude and longitude from nodes
                    lat_lon_pairs = [(node['lat'], node['lon']) for node in node_segment if 'lat' in node and 'lon' in node]
                    
                    if len(lat_lon_pairs) < 2:
                        # Skip if insufficient points
                        continue
                    
                    # Convert geographic coordinates to image coordinates
                    image_coords = []
                    for lat, lon in lat_lon_pairs:
                        x = (lon - min_lon) / lon_span * img_size
                        y = (1 - (lat - min_lat) / lat_span) * img_size
                        image_coords.append((x, y))
                    
                    # Check if way is closed (first and last node are the same)
                    is_closed = False
                    nodes = element.get('nodes')
                    if len(node_segment) >= 2:
                        first_node_id = nodes[0]
                        last_node_id = nodes[-1]
                        is_closed = first_node_id == last_node_id
                        
                    try:
                        if is_closed and len(image_coords) >= 3:
                            # For closed ways, create a Shapely polygon
                            polygon = ShapelyPolygon(image_coords)
                            
                            # Check intersection with each grid cell
                            for row in range(num_grids):
                                for col in range(num_grids):
                                    # Define the grid cell as a polygon
                                    cell_left = col * cell_size
                                    cell_top = row * cell_size
                                    cell_right = (col + 1) * cell_size
                                    cell_bottom = (row + 1) * cell_size
                                    
                                    grid_cell = ShapelyPolygon([
                                        (cell_left, cell_top),
                                        (cell_right, cell_top),
                                        (cell_right, cell_bottom),
                                        (cell_left, cell_bottom)
                                    ])
                                    
                                    # If the element intersects with this cell, add it to the mask
                                    if polygon.intersects(grid_cell):
                                        cell_index = index_from_xy(row, col, num_grids)
                                        if cell_index not in mask:
                                            mask.append(cell_index)
                        else:
                            # For open ways, process as a LineString
                            if len(image_coords) < 2:
                                continue
                                
                            # Create a LineString for each pair of consecutive points
                            for i in range(len(image_coords) - 1):
                                line = ShapelyLineString([image_coords[i], image_coords[i + 1]])
                                
                                # Check intersection with each grid cell
                                for row in range(num_grids):
                                    for col in range(num_grids):
                                        # Define the grid cell as a polygon
                                        cell_left = col * cell_size
                                        cell_top = row * cell_size
                                        cell_right = (col + 1) * cell_size
                                        cell_bottom = (row + 1) * cell_size
                                        
                                        grid_cell = ShapelyPolygon([
                                            (cell_left, cell_top),
                                            (cell_right, cell_top),
                                            (cell_right, cell_bottom),
                                            (cell_left, cell_bottom)
                                        ])
                                        
                                        # If the line intersects with this cell, add it to the mask
                                        if line.intersects(grid_cell):
                                            cell_index = index_from_xy(row, col, num_grids)
                                            if cell_index not in mask:
                                                mask.append(cell_index)
                    except Exception as e:
                        print(f"Warning: Failed to process way segment: {str(e)}")
                        continue
            except Exception as e:
                print(f"Warning: Failed to process way element {element_id}: {str(e)}")
                continue
        
        # Store the mask if there are any cell indices
        if mask:
            masks[tag_key] = mask
    
    return masks

def visualize_grid_masks(masks, img_size, cell_size, image_path, output_path = None):
    '''
    Creates a visualization of OSM element grid masks on top of a satellite image.
    
    Input:
    - masks: dict. Keys are string representations of element tags, values are lists of grid cell indices.
    - img_size: int. The size of the image in pixels (assuming square).
    - cell_size: int. The size of each grid cell in pixels.
    - image_path: str. Path to the satellite image to use as background.
    - output_path: str. Path where the visualization will be saved.
    
    Output:
    - Saves the visualization to the specified output path.
    '''
    
    # Create helper function to convert linear index to row, col
    def xy_from_index(idx, grid_size):
        row = idx // grid_size
        col = idx % grid_size
        return row, col
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Load and display the satellite image
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((img_size, img_size))
        img = np.array(img)
        ax.imshow(img)
    else:
        # If image doesn't exist, create a blank white background
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_facecolor('white')
        print(f"Warning: Image not found at {image_path}, using blank background")
    
    # Draw grid lines
    num_grids = img_size // cell_size
    for i in range(num_grids + 1):
        ax.axhline(i * cell_size, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(i * cell_size, color='gray', linestyle='-', alpha=0.3)
    
    # Generate colors for each OSM element type
    colors = list(mcolors.TABLEAU_COLORS.values())
    class_colors = {}
    
    # Create a more readable version of tag keys for the legend
    readable_tags = {}
    for tag_key in masks.keys():
        try:
            # Try to convert string representation of dict to actual dict
            tag_dict = ast.literal_eval(tag_key)
            
            # Create a readable version (e.g., "river, waterway")
            readable_version = ", ".join([f"{v}" if k == v else f"{k}={v}" for k, v in tag_dict.items()])
            readable_tags[tag_key] = readable_version
        except:
            # If parsing fails, use the original string
            readable_tags[tag_key] = tag_key
    
    # Assign colors to each tag
    for i, tag_key in enumerate(masks.keys()):
        class_colors[tag_key] = colors[i % len(colors)]
    
    # Draw each element's grid cells
    for tag_key, indices in masks.items():
        color = class_colors[tag_key]
        
        for idx in indices:
            row, col = xy_from_index(idx, num_grids)
            
            # Create rectangle for this grid cell
            rect = Rectangle(
                (col * cell_size, row * cell_size),
                cell_size, cell_size,
                facecolor=color,
                edgecolor='none',
                alpha=0.5
            )
            ax.add_patch(rect)
    
    # Add title and legend
    ax.set_title("OpenStreetMap Elements Grid Visualization")
    
    # Create legend entries
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.5) 
        for color in class_colors.values()
    ]
    legend_labels = [readable_tags[tag] for tag in class_colors.keys()]
    
    # Add legend outside the plot area
    ax.legend(
        legend_handles, 
        legend_labels, 
        loc='center left', 
        bbox_to_anchor=(1.05, 0.5),
        fontsize='small'
    )

    if not output_path:
        output_path = os.path.join(OUTPUT_VIS_DIR, f"final_nuovo_{os.path.basename(image_path)}")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    data_folder = "/media/data_fast/Riccardo/OpenStreetCLIP_dataset"
    # file_name = "38.10856189889793_-85.8059195342252_38.11066731571431_-85.80324376118514"
    file_name = "25.012971027234663_-111.67307963628139_25.015076444051047_-111.67075628335289"

    PATCH_SIZE = 32
    IMAGE_SIZE = 224

    print(f"Analyzing file: {file_name}")

    masks = extract_masks_from_osm_image(
        f"{data_folder}/osm_metadata/{file_name}.json", IMAGE_SIZE, PATCH_SIZE
    )
    print_dictionary(masks) 
    visualize_grid_masks(masks, IMAGE_SIZE, PATCH_SIZE, f"{data_folder}/images/{file_name}.png", f"temp/retrieve_ids_images/final_nuovo_{file_name}.png")
    print("\n"*5)    


# if __name__ == "__main__":
#     data_folder = "/media/data_fast/Riccardo/osm_data_rich"
#     json_files = os.listdir(f"{data_folder}/postprocessed_osm_data")
#     PATCH_SIZE = 16
#     IMAGE_SIZE = 224

#     for file in json_files:
#         print(f"Analyzing file: {file}")
#         jpg_image_file = file.replace("json", "png")

#         masks = extract_masks_from_osm_image(f"{data_folder}/postprocessed_osm_data/{file}", IMAGE_SIZE, PATCH_SIZE)
#         print_dictionary(masks)

#         visualize_grid_masks(masks, IMAGE_SIZE, PATCH_SIZE, f"{data_folder}/images/{jpg_image_file}", f"temp/retrieve_ids_images/final_nuovo_{jpg_image_file}")
#         print("\n"*5)

#         True, f"{data_folder}/images/{jpg_image_file}"

# FIXME
# Switch from the definition of the grid dimensions to the definition of the small patch size. Use only the patch size as an argument to the function. (Alessio)
# Filter the tags (Riccardo)
# Sometimes the function fail with this error "TypeError: list indices must be integers or slices, not str" (Alessio)