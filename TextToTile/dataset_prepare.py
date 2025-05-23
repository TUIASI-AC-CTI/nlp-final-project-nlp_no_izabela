import os
import pandas as pd
from shutil import copyfile

tiles_dir = "data/tiles"
captions_file = "data/tile_captions.csv"
output_dir = "data/dataset_lora"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(captions_file)
for _, row in df.iterrows():
    img = row["filename"]
    caption = row["caption"]
    img_path = os.path.join(tiles_dir, img)
    copyfile(img_path, os.path.join(output_dir, img))
    with open(os.path.join(output_dir, img.replace(".png", ".txt")), "w", encoding="utf-8") as f:
        f.write(caption)

print("✅ Dataset pregătit în: data/dataset_lora")
