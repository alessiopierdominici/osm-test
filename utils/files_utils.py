import os

def get_image_and_json_files(file_path):
    image_files = []
    json_files = []
    images_folder = os.path.join(file_path, "images")
    json_folder = os.path.join(file_path, "postprocessed_osm_data")
    
    # Verifica che le cartelle di immagini e JSON esistano
    if not os.path.exists(images_folder) or not os.path.exists(json_folder):
        return image_files, json_files
    
    # Ottieni i file immagine dalla sotto-cartella "images"
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))
                
    # Ottieni i file JSON dalla sotto-cartella "postprocessed_osm_data"
    for root, dirs, files in os.walk(json_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    return image_files, json_files


def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key}: {value}')