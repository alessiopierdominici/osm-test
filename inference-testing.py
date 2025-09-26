import torch
import os
import clip
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
from retrieves_ids import extract_masks_from_osm_image, print_dictionary, visualize_grid_masks
from utils.general_utils import prepare_tags
import logging
logging.basicConfig(level=logging.INFO)
import shutil


# MODEL_NAME = "/media/data_fast/Alessio/OpenStreetCLIP/checkpoints/train_osm_new_dataset_checkpoints/20250325_010743_Task:new_dataset_mpa_32_train_OSM_filtro_20250325_005246_OrigClip:False_Opt:Adafactor_AUG:False_DS:OSM_LR:5e-05_E:100_BS:96/BestModel_20250325_010743_Task:new_dataset_mpa_32_train_OSM_filtro_20250325_005246_OrigClip:False_Opt:Adafactor_AUG:False_DS:OSM_LR:5e-05_E:100_BS:96_epoch_7.pt"
MODEL_NAME = "/media/data_fast/Alessio/OpenStreetCLIP/checkpoints/finetuned_4_datasets_checkpoints_gradient_accumulation/20250401_102335_Task:FT_OrigClip:False_Opt:Adafactor_AUG:False_AUGN:False_NAUGIM:-1_DS:4_LR:0.0005_E:100_BS:256/BestModel_20250401_102335_Task:FT_OrigClip:False_Opt:Adafactor_AUG:False_AUGN:False_NAUGIM:-1_DS:4_LR:0.0005_E:100_BS:256_epoch_2.pt"


RETURN_CLS_FOR_OSM = False
IMAGE_SIZE = 224
PATCH_SIZE = 32
CUDA_DEVICE = 1
USE_AUGMENTATION = False
TESTING_FOLDER = "/media/data_fast/Riccardo/OpenStreetCLIP_dataset"

# Visualization parameters
DISPLAY_PROB_VALUES = True     # Show the probability values on each grid cell
APPLY_HEATMAP = True           # Apply color gradient (blue to red) to cells based on probability
HIGHLIGHT_TOP_K = 4           # Number of top cells to highlight in lime green (0 to disable)

# TRAIN_SAMPLES = [
#     "38.10856189889793_-85.8059195342252_38.11066731571431_-85.80324376118514",
#     "37.29441387111428_-81.21119171538268_37.296519287930664_-81.20854509609197",
#     "38.53971102334245_-75.30497654053062_38.54181644015883_-75.3022848044875",
# ]

# VAL_SAMPLES = [
#     "38.10856189889793_-85.8059195342252_38.11066731571431_-85.80324376118514",
#     "37.29441387111428_-81.21119171538268_37.296519287930664_-81.20854509609197",
#     "38.53971102334245_-75.30497654053062_38.54181644015883_-75.3022848044875",
# ]

TESTING_SAMPLES = [
    "35.653787258111635_-97.27733806211333_35.65589267492801_-97.27474695640544",
    "30.702060569739825_-94.07526500601153_30.7043602463478_-94.07259041503917",
    "43.724508809031434_-116.36734351178_43.72661422584781_-116.36443003623476",
    "34.41479209905936_-109.85289420095441_34.41709177566734_-109.85010656639642",
    "32.91996200314398_-100.4642093922426_32.922261679751955_-100.46146978926669",
    "32.23449907862545_-80.84958600138242_32.23679875523342_-80.84686726227477",
    "46.237673582425046_-81.06472331548966_46.23977899924142_-81.06167923818893",
    "45.03055126196388_-91.0167903755445_45.032850938571855_-91.01353634077819",
    "33.46137987482635_-106.40580826493142_33.463485291642726_-106.40328450682702",
    "38.878090405576465_-91.96186414443446_38.88019582239285_-91.95915955815529",
    "41.82964019707207_-69.9422264173397_41.83174561388845_-69.93940085788012",
    "42.22459593435455_-122.02112369553116_42.22689561096252_-122.01801813405923",
    "39.04608464483508_-104.68013643446805_39.04819006165146_-104.67742550803108",
    "34.92895405200022_-87.00065857116398_34.9310594688166_-86.99809056390366",
    "32.07499246252987_-96.43993929640622_32.077292139137846_-96.43722531129903",
    "40.20911791144799_-76.62116900750543_40.21122332826437_-76.61841212532545",
    "48.51934537336952_-115.76386652259039_48.521645049977494_-115.76039453749964",
    "29.98471606150943_-97.81222135122928_29.986821478325815_-97.80979060300999",
    "39.25592747969859_-83.07701302031172_39.258032896514976_-83.07429391671795",
    "39.587288230922965_-117.15013158871801_39.58958790753094_-117.14714748320617",
    "40.44046035803437_-91.72936542623391_40.44276003464235_-91.72634377986039",
    "43.511164605531285_-79.72363351814676_43.51327002234767_-79.72073036234067",
    "45.129515089058245_-71.33875400668629_45.13162050587462_-71.33576964141207",
    "39.7658490486163_-94.35940855740193_39.76795446543268_-94.35666950627146",
    "38.864942286384405_-83.9669726244689_38.86724196299238_-83.96401907533169",
    "40.84651118669317_-77.77337426749165_40.848616603509555_-77.7705909516935",
    "42.1233631610106_-76.10231200164033_42.125468577826986_-76.0994733804595",
    "41.342274976033025_-80.97972439285508_41.3443803928494_-80.97692008158225",
    "39.21421768966521_-87.14566902735251_39.21651736627319_-87.14270084074238",
    "29.35146650048235_-81.07227947474348_29.353571917298737_-81.06986393254257",
    "34.57841733036517_-86.29155612853275_34.58052274718155_-86.28899893407825",
    "34.76206533182806_-84.50798579348522_34.76436500843604_-84.50518648324045",
    "30.433771001192817_-97.65089562879236_30.4358764180092_-97.64845371045801",
    "36.74554830790288_-76.54386114539541_36.74765372471926_-76.541233577758",
    "33.02453539770606_-94.20217837143797_33.02664081452244_-94.19966725515704",
    "41.254226878231314_-110.96716623435398_41.25652655483929_-110.96410725042335",
]

# TESTING_SAMPLES = [

#     "41.60330248026733_-80.45034592020775_41.605407897083715_-80.44753029388343",

#     "39.445479813619244_-108.02062300544496_39.44777949022722_-108.01764498584872",
#     "40.50499523759504_-95.21783710407723_40.50710065441142_-95.21506801046837",
#     "31.252866360430644_-101.08215769044237_31.25497177724703_-101.07969488968139",
#     "31.252866360430644_-101.08215769044237_31.25497177724703_-101.07969488968139",
#     "44.42730149076294_-87.89214948858395_44.42960116737092_-87.88892921749394",


#     # "38.756291136271415_-75.5973698980048_38.75839655308779_-75.5946700132278",
#     # "42.6285636375373_-76.30090473123677_42.63086331414527_-76.29777909177982",
#     # "41.8792331374366_-103.67136050932399_41.88133855425298_-103.66853275814",
#     # "37.568656410473345_-119.26680781874829_37.57076182728973_-119.26415148458062",
#     # "41.42138724760042_-74.15596172315091_41.423492664416806_-74.15315399827742",
#     # "35.55011649987143_-82.6492273310529_35.55222191668781_-82.64663951196019",
#     # # 38.756291136271415_-75.5973698980048_38.75839655308779_-75.5946700132278.json
#     # # 42.6285636375373_-76.30090473123677_42.63086331414527_-76.29777909177982.json
#     # # 41.8792331374366_-103.67136050932399_41.88133855425298_-103.66853275814.json
#     # "38.756291136271415_-75.5973698980048_38.75839655308779_-75.5946700132278",
#     # "42.6285636375373_-76.30090473123677_42.63086331414527_-76.29777909177982",
#     # "41.8792331374366_-103.67136050932399_41.88133855425298_-103.66853275814",
#     # "48.462426571960336_-98.70933220631854_48.46453198877672_-98.70615715596227",
#     # "45.6221064241224_-92.44301604981189_45.624211840938784_-92.44000557244063",
#     # "46.4499380097493_-122.87214892949343_46.45223768635728_-122.86881109951251",
#     # "38.756291136271415_-75.5973698980048_38.75839655308779_-75.5946700132278",
# ]

# 37.568656410473345_-119.26680781874829_37.57076182728973_-119.26415148458062.json
# 41.42138724760042_-74.15596172315091_41.423492664416806_-74.15315399827742.json
# 35.55011649987143_-82.6492273310529_35.55222191668781_-82.64663951196019.json

# 48.462426571960336_-98.70933220631854_48.46453198877672_-98.70615715596227.json
# 45.6221064241224_-92.44301604981189_45.624211840938784_-92.44000557244063.json
# 46.4499380097493_-122.87214892949343_46.45223768635728_-122.86881109951251.json

# 38.756291136271415_-75.5973698980048_38.75839655308779_-75.5946700132278.json
# 42.6285636375373_-76.30090473123677_42.63086331414527_-76.29777909177982.json
# 41.8792331374366_-103.67136050932399_41.88133855425298_-103.66853275814.json

DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

OUTPUT_VIS_DIR = "temp/inference-testing/"

if os.path.exists(OUTPUT_VIS_DIR):
    shutil.rmtree(OUTPUT_VIS_DIR)
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

def load_image_and_json(sample_name, folder):
    image_path = os.path.join(f"{folder}/images/", f"{sample_name}.png")
    json_path = os.path.join(f"{folder}/osm_metadata/", f"{sample_name}.json")

    print(f"Loading image from: {image_path}")
    print(f"Loading JSON from: {json_path}")

    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        return None, None

    if not os.path.exists(json_path):
        logging.error(f"JSON not found: {json_path}")
        return None, None

    image = Image.open(image_path).convert("RGB")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    return image, json_data

def infer_on_sample(sample_name, model, preprocess, device, output_folder_sample):
    image, json_data = load_image_and_json(sample_name, TESTING_FOLDER)
    if image is None:
        return
    
    img_to_show = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_to_show = np.array(img_to_show)

    # Save image with grid divisions
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_to_show)
    
    # Draw the grid lines manually to ensure they're visible
    for i in range(0, IMAGE_SIZE + 1, PATCH_SIZE):
        ax.axhline(y=i, color='white', linewidth=0.5, alpha=0.8)  # Horizontal lines
        ax.axvline(x=i, color='white', linewidth=0.5, alpha=0.8)  # Vertical lines
    
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)  # Flip y-axis to match image coordinates
    
    plt.title(f"Grid Division - {sample_name}")
    plt.tight_layout()
    
    # Save the grid image
    grid_save_path = os.path.join(output_folder_sample, f"{sample_name}_grid_only.png")
    plt.savefig(grid_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"Saved grid image to {grid_save_path}")
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    patches = []
    for _, element in json_data.items():
        tags = element.get('tags', {}).copy()
        tags.pop("position", None)
        patches.append(str(tags))

    list_tags = prepare_tags(patches)
    list_tags = list(set(list_tags))

    tokenized_tags = clip.tokenize(list_tags, truncate=True).to(device)

    with torch.no_grad():
        image_features, text_features = model(image_tensor, tokenized_tags)
    
    logit_scale = model.logit_scale.exp()
    img_feat = image_features[0]
    
    grid_size = IMAGE_SIZE // PATCH_SIZE
    
    for tag_idx, tag in enumerate(list_tags):
        txt_feat = text_features[tag_idx]
        patch_similarities = logit_scale * (img_feat @ txt_feat.T)
        patch_probs = torch.softmax(patch_similarities, dim=0).detach()
        activation_map = patch_probs.cpu().numpy().reshape(grid_size, grid_size)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_to_show)
        
        top_k_indices = []
        if HIGHLIGHT_TOP_K > 0:
            flat_indices = np.argsort(activation_map.flatten())[-HIGHLIGHT_TOP_K:][::-1]
            for idx in flat_indices:
                row, col = idx // grid_size, idx % grid_size
                top_k_indices.append((row, col))
        
        if APPLY_HEATMAP:
            im = ax.imshow(activation_map, cmap='jet', alpha=0.5, 
                           interpolation='nearest', 
                           extent=[0, IMAGE_SIZE, IMAGE_SIZE, 0],
                           vmin=0, vmax=1)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Activation Strength')
            
        ax.set_xticks(np.arange(0, IMAGE_SIZE+1, PATCH_SIZE))
        ax.set_yticks(np.arange(0, IMAGE_SIZE+1, PATCH_SIZE))
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        for i in range(grid_size):
            for j in range(grid_size):
                x_center = j * PATCH_SIZE + PATCH_SIZE // 2
                y_center = i * PATCH_SIZE + PATCH_SIZE // 2
                
                x_min = j * PATCH_SIZE
                y_min = i * PATCH_SIZE
                
                # if (i, j) in top_k_indices:
                #     rect = plt.Rectangle((x_min, y_min), PATCH_SIZE, PATCH_SIZE, 
                #                          fill=False, edgecolor='lime', linewidth=2)
                #     ax.add_patch(rect)
                
                if DISPLAY_PROB_VALUES:
                    if (i, j) in top_k_indices:
                        text_color = 'lime'
                    else:
                        text_color = 'black'
                    prob_text = f"{activation_map[i, j]:.2f}"

                    ax.text(x_center, y_center, prob_text, 
                            ha='center', va='center', 
                            color=text_color, fontsize=8)
        
        plt.title(f"Tag: {tag}")
        
        plt.tight_layout()
        save_path = os.path.join(output_folder_sample, f"{sample_name}_tag_{tag_idx}.png")
        plt.savefig(save_path)
        plt.close()
        
        logging.info(f"Saved visualization for tag {tag_idx}: {tag} to {save_path}")
        
        flat_idx = np.argsort(activation_map.flatten())[-HIGHLIGHT_TOP_K:][::-1]
        for i, idx in enumerate(flat_idx):
            row, col = idx // grid_size, idx % grid_size
            logging.info(f"  Top {i+1} activation: Patch ({row},{col}) - Value: {activation_map[row, col]:.4f}")

def main():
    from create_model import _load_osm_clip
    model, preprocess = _load_osm_clip(USE_AUGMENTATION, name=MODEL_NAME, 
                                       device=DEVICE, return_cls=RETURN_CLS_FOR_OSM)
    logging.info(f"Model loaded: {MODEL_NAME}")
    model.eval()

    for sample in TESTING_SAMPLES:
        logging.info(f"Processing sample: {sample}")
        output_folder_sample = f"{OUTPUT_VIS_DIR}/{sample}"
        os.makedirs(output_folder_sample, exist_ok=True)

        infer_on_sample(sample, model, preprocess, DEVICE, output_folder_sample)

        masks = extract_masks_from_osm_image(
            f"{TESTING_FOLDER}/osm_metadata/{sample}.json", IMAGE_SIZE, PATCH_SIZE
        )
        print_dictionary(masks) 
        visualize_grid_masks(masks, IMAGE_SIZE, PATCH_SIZE, f"{TESTING_FOLDER}/images/{sample}.png", f"{output_folder_sample}/GT_{sample}.png")

    logging.info(f"Visualizations saved to {OUTPUT_VIS_DIR}")

if __name__ == "__main__":
    main()
