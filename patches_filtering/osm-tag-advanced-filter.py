import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import Counter

from utils.general_utils import get_formatted_datetime

# =============== CONFIGURATION CONSTANTS ===============

# Keys to be excluded from tags (will be removed along with their values)
# KEYS_TO_EXCLUDE = [
#     '4wd_only', 'voltage', 'frequency', 'start_date',
#     'electrified', 'HFCS', 'ref', 'source', 'attribution', 'created_by',
#     'timestamp', 'version', 'note', 'fixme', 'todo', 'gauge', 'topspeed', 'cables', 'opening_hours',
#     'destination', 'brand', 'network', 'network:short', 'species', 'designation',# have a lot of proper names
#     'width', 'length'
#     'workrules', 'maxheight', 'maxweight', 'maxwidth', 'maxaxleload', 'maxstay', 'maxspeed:advisory', 'maxspeed:advisory:forward', 'maxspeed:advisory:backward', 'maxspeed:advisory:conditional', 'maxspeed:advisory:conditional:forward',
#     'direction','genus','fee', 'iata', 'Creek', 'internet_access', 'internet_access:fee',
#     'from', 'to', 'iucn_level', 'manufacturer', 'diocese','railway:track_ref'
# ]

# # Forbidden substrings (if these appear in any key or value, that pair will be removed)
# FORBIDDEN_SUBSTRINGS = [
#     '@', 'mph', 'km/h', 'kmh', 'kph', 'maxspeed', 'minspeed', 'speed', 'lanes', 'http' , 'date' ,'colour',
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', # Numbers
#     'workrules', '&',
# ]

# Keys to be excluded from tags (will be removed along with their values)
KEYS_TO_EXCLUDE = [
    "zoo",
    "width",
    "length",
    "gauge",
    "check_date",
    "maxspeed",
    "minspeed",
    "lanes",
    "maxspeed:advisory",
    "maxspeed:advisory:forward",
    "maxspeed:advisory:backward",
    "maxspeed:advisory:conditional",
    "maxspeed:advisory:conditional:forward",
    "motor_vehicle:conditional",
    "internet_access",
    "internet_access:fee",
    "brand",
    "opening_hours",
    "opening_hours:signed",
    "building:levels",
    "start_date",
    "end_date",
    "survey:date",
    "frequency",
    "voltage",
    "population",
    "destination",
    "isced:level",
    "railway:track_ref",
    "level",
    "protect_class",
    "roof:colour",
    "rooms",
    "opening_date",
    "layer",
    "man_made",
    "heritage",
    "roof:levels",
    "building:min_level",
    "cables",
    "direction",
    "camera:type",
    "winter_service:priority",
    "placement",
    "workrules",
    "capacity",
    "capacity:disabled",
    "step_count",
    "incline",
    "maxspeed:hgv",
    "maxspeed:conditional",
    "network",
    "network:wikidata",
    "mtb:scale:imba",
    "mtb:scale",
    "building:flats",
    "cuisine",
    "delivery",
    "takeaway",
    "drive_through",
    "est_width",
    "depth",
    "designation",
    "distance",
    "handicap",
    "par",
    "railway:position",
    "maxweight",
    "hoops",
    "lit",
    "manufacturer",
    "rotor:diameter",
    "species",
    "seats",
    "backrest",
    "tourism",
    "payment:notes:denominations",
    "location",
    "artwork_type",
    "office",
    "fee",
    "religion",
    "denomination",
    "maxspeed:backward",
    "maxweight",
    "motor_vehicle",
    "from",
    "to",
    "F",
    "F1",
    "seats",
    "screen",
    "healthcare:speciality",
    "healthcare",
    "maxstay",
    "diocese",
    "collection_times",
    "crop",
    "area",
    "network:short",
    "taxi",
    "voltage:primary",
    "voltage:secondary",
    "line_attachment",
    "dance:style",
    "dance:teaching",
    "devices",
    "aerialway:occupancy",
    "iata",
    "internet_access:ssid",
    "traffic_sign",
    "genus",
    "leaf_cycle",
    "leaf_type",
    "artist:wikidata",
    "maxspeed:forward",
    "model",
    "maxspeed:variable",
    "maxspeed:hgv:backward",
]

# Forbidden substrings (if these appear in any key or value, that pair will be removed)
FORBIDDEN_SUBSTRINGS = ["http", "#", "COLUMBIA", "VELO"]

# Whether to create a backup before overwriting output folder
CREATE_BACKUP = True


def contains_forbidden_substring(text, forbidden_substrings):
    """Check if a string contains any forbidden substring"""
    if not isinstance(text, str):
        return False

    for substring in forbidden_substrings:
        if substring in text:
            return True
    return False


def filter_tag_dict(tag_dict, excluded_keys, forbidden_substrings):
    """
    Filter a tag dictionary by:
    1. Removing keys in the excluded list
    2. Removing key-value pairs where key or value contains a forbidden substring

    Returns: Filtered dictionary
    """
    filtered_dict = {}

    for key, value in tag_dict.items():
        # Skip excluded keys
        if key in excluded_keys:
            continue

        # Skip if key or value contains a forbidden substring
        if contains_forbidden_substring(
            key, forbidden_substrings
        ) or contains_forbidden_substring(value, forbidden_substrings):
            continue

        # Keep this key-value pair
        filtered_dict[key] = value

    return filtered_dict


def filter_tags(input_folder, output_folder, force=False):
    """
    Filter OSM tags by removing excluded keys and numeric values

    Args:
        input_folder: Path to folder containing OSM JSON files
        output_folder: Path to save filtered JSON files
        force: Whether to overwrite output folder without prompting
    """
    print(f"Filtering OSM tags from {input_folder} to {output_folder}")
    print(f"Keys to exclude: {KEYS_TO_EXCLUDE}")
    print(f"Forbidden substrings: {FORBIDDEN_SUBSTRINGS}")

    # Handle output folder
    if os.path.exists(output_folder) and not force:
        response = input(
            f"Output folder {output_folder} already exists. Overwrite? (y/n): "
        )
        if response.lower() != "y":
            print("Aborting.")
            return

    # Create backup if requested
    if os.path.exists(output_folder) and CREATE_BACKUP:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = f"{output_folder}_backup_{timestamp}"
        print(f"Creating backup of existing output folder to {backup_folder}")
        import shutil

        shutil.copytree(output_folder, backup_folder)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Stats for reporting
    stats = {
        "total_files": 0,
        "total_tags_processed": 0,
        "total_tags_kept": 0,
        "total_tags_filtered": 0,
        "filter_reasons": Counter(),
    }

    # Get all JSON files
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files to filter")
    stats["total_files"] = len(json_files)

    # Process each file
    for filename in tqdm(json_files):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with open(input_path, "r") as f:
                data = json.load(f)

            filtered_data = {}

            for tag_str, patches in data.items():
                stats["total_tags_processed"] += 1

                # Skip empty tags
                if tag_str == "{}" or not tag_str:
                    stats["filter_reasons"]["empty_tag"] += 1
                    continue

                # Process the tag
                try:
                    # Convert tag string to dictionary
                    tag_dict = json.loads(tag_str.replace("'", '"'))

                    # Apply filtering
                    filtered_tag_dict = filter_tag_dict(
                        tag_dict, KEYS_TO_EXCLUDE, FORBIDDEN_SUBSTRINGS
                    )

                    # Skip if no keys remain after filtering
                    if not filtered_tag_dict:
                        stats["filter_reasons"]["no_keys_remain"] += 1
                        stats["total_tags_filtered"] += 1
                        continue

                    # Convert filtered dict back to string and add to output
                    filtered_tag_str = str(filtered_tag_dict)
                    if filtered_tag_str in filtered_data:
                        filtered_data[filtered_tag_str].extend(patches)
                        # Remove duplicates
                        filtered_data[filtered_tag_str] = list(
                            set(filtered_data[filtered_tag_str])
                        )
                    else:
                        filtered_data[filtered_tag_str] = patches

                    stats["total_tags_kept"] += 1

                except Exception as e:
                    stats["filter_reasons"]["parse_error"] += 1
                    stats["total_tags_filtered"] += 1
                    continue

            if filtered_data:
                # Save filtered data
                with open(output_path, "w") as f:
                    json.dump(filtered_data, f)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Calculate total filtered tags
    stats["total_tags_filtered"] = (
        stats["total_tags_processed"] - stats["total_tags_kept"]
    )

    # Print summary statistics
    print("\nFiltering Summary:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total tags processed: {stats['total_tags_processed']}")
    print(
        f"Tags kept: {stats['total_tags_kept']} ({stats['total_tags_kept']/max(1, stats['total_tags_processed'])*100:.1f}%)"
    )
    print(
        f"Tags filtered: {stats['total_tags_filtered']} ({stats['total_tags_filtered']/max(1, stats['total_tags_processed'])*100:.1f}%)"
    )
    print("\nFilter reasons:")
    for reason, count in stats["filter_reasons"].most_common():
        print(f"  {reason}: {count} tags")

    # Save filtering report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "keys_to_exclude": KEYS_TO_EXCLUDE,
            "forbidden_substrings": FORBIDDEN_SUBSTRINGS,
            "input_folder": input_folder,
            "output_folder": output_folder,
        },
        "statistics": stats,
    }

    report_path = (
        f"temp/osm-tag-filter-report/{get_formatted_datetime()}_filtering_report.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFiltering complete! Filtered files saved to {output_folder}")
    print(f"Detailed report saved to '{report_path}'")

    # print how many files are in both folders
    # input_files = os.listdir(args.input_folder)
    # output_files = os.listdir(args.output_folder)
    input_files = os.listdir(input_folder)
    output_files = os.listdir(output_folder)
    print(f"Input files: {len(input_files)}")
    print(f"Output files: {len(output_files)}")
    print(f"Common files: {len(set(input_files).intersection(output_files))}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter OSM tags in JSON files")
    # parser.add_argument("input_folder", help="Folder containing OSM JSON files")
    # parser.add_argument("output_folder", help="Folder to save filtered JSON files")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of output folder without prompting",
    )

    args = parser.parse_args()

    # filter_tags(args.input_folder, args.output_folder, args.force)

    input_folder = "/media/data_fast/Riccardo/OpenStreetCLIP_final_dataset/masks/32"
    output_folder = "/media/data_fast/Riccardo/OpenStreetCLIP_final_dataset/masks/32_f2"
    filter_tags(input_folder, output_folder, args.force)
