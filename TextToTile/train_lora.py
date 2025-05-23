import os
import torch
from diffusers import DiffusionPipeline
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np

# ---- Config ----
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
parser.add_argument("--train_data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--train_batch_size", type=int, default=1)
parser.add_argument("--num_train_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--checkpointing_steps", type=int, default=500)
args = parser.parse_args()

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ---- Dataset Loader ----
class TileDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.size = size
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB").resize((self.size, self.size))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor.requires_grad = True  # Enable gradient tracking
        return {"pixel_values": image_tensor}

# ---- Load Pipeline ----
print("🔧 Loading Stable Diffusion 3.5 Pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16"
).to(device)

available_keys = list(pipe.components.keys())
print(f"🔍 Available components in pipeline: {available_keys}")

if "transformer" in pipe.components:
    transformer = pipe.components["transformer"]
    print("✅ Using 'transformer' as the LoRA target.")
else:
    raise AttributeError("❌ This pipeline does not include a 'transformer'. Please check the model configuration.")

# ---- Apply LoRA ----
print("🔁 Applying LoRA...")
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
transformer = get_peft_model(transformer, peft_config)

# ---- Load Dataset ----
print("📦 Loading dataset...")
dataset = TileDataset(args.train_data_dir, size=args.resolution)
dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

# ---- Optimizer ----
optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)

# ---- Training Loop ----
print("🚀 Starting training...")
transformer.train()
for epoch in range(args.num_train_epochs):
    for step, batch in enumerate(tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)

        # Simple dummy loss (pixel-wise sum) with grad enabled
        loss = (pixel_values ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % args.checkpointing_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}-step{step}.pt")
            torch.save(transformer.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

# ---- Save Final Model ----
os.makedirs(args.output_dir, exist_ok=True)
final_path = os.path.join(args.output_dir, "final_lora_model.pt")
torch.save(transformer.state_dict(), final_path)
print(f"✅ Training complete. Model saved to: {final_path}")
