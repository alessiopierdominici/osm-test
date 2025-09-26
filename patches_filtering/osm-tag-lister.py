import os
import json

def extract_keys_from_json(folder_path, output_file):
    all_strings = set()

    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    if isinstance(data, dict):
                        all_strings.update(data.keys())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Save all extracted strings to the output file
    with open(output_file, "w", encoding="utf-8") as out_file:
        for string in sorted(all_strings):
            out_file.write(string + "\n")

    print(f"Extraction complete. Strings saved in {output_file}")

# Example usage
folder_path = "/media/data_fast/Riccardo/OpenStreetCLIP_final_dataset/masks/32_f2"  # Change this to your folder path
output_file = "osm_tags_output.log"
extract_keys_from_json(folder_path, output_file)
