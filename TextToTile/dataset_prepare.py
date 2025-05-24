import os
import pandas as pd
from shutil import copyfile

tiles_dir = "data/tiles"
captions_file = "data/tile_captions.csv"
output_dir = "data/dataset_lora"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(captions_file)
for idx, row in df.iterrows():
    original_img = row["filename"]
    caption = row["caption"]

    img_path = os.path.join(tiles_dir, original_img)

    new_img_name = f"tile_{idx}.png"
    new_txt_name = f"tile_{idx}.txt"

    copyfile(img_path, os.path.join(output_dir, new_img_name))
    with open(os.path.join(output_dir, new_txt_name), "w", encoding="utf-8") as f:
        f.write(caption)

print("✅ Dataset ready: data/dataset_lora")
