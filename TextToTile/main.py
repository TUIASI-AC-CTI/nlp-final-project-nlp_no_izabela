import os
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import uuid
import numpy as np

# ---- Config ----
class Args:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"
    output_dir = "lora_output"
    resolution = 64

args = Args()

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ---- Load base model pipeline ----
print("🔧 Loading base model pipeline...")
base_generator = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    revision="fp16" if torch.cuda.is_available() else "main",
    use_auth_token=True
).to(device)

# ---- Load fine-tuned model pipeline ----
print("🔧 Loading fine-tuned model pipeline...")
fine_tuned_pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16"
).to(device)

if "transformer" not in fine_tuned_pipe.components:
    raise AttributeError("❌ Fine-tuned pipeline does not include a 'transformer'.")

transformer = fine_tuned_pipe.components["transformer"]
print("✅ Applying LoRA to fine-tuned transformer...")

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
transformer = get_peft_model(transformer, peft_config)

state_dict_path = os.path.join(args.output_dir, "final_lora_model.pt")
transformer.load_state_dict(torch.load(state_dict_path, map_location=device))
fine_tuned_pipe.components["transformer"] = transformer

# ---- Tkinter GUI ----
root = tk.Tk()
root.title("Text-to-Tile Generator - Base vs Fine-Tuned")
root.geometry("1024x800")

prompt_label = tk.Label(root, text="Enter Tile Description:")
prompt_label.pack(pady=10)

prompt_entry = tk.Entry(root, width=60)
prompt_entry.pack(pady=5)
prompt_entry.insert(0, "A top-down tile of a forest floor")

canvas = tk.Canvas(root, width=1024, height=512)
canvas.pack(pady=10)

image_tk_1 = image_tk_2 = None


def generate():
    global image_tk_1, image_tk_2

    prompt = prompt_entry.get()
    if not prompt:
        messagebox.showerror("Error", "Please enter a description.")
        return

    with torch.autocast(device_type=device.type):
        base_img = base_generator(prompt, height=512, width=512, num_inference_steps=30).images[0]
        fine_img = fine_tuned_pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]

    base_img_resized = base_img.resize((512, 512), Image.LANCZOS)
    fine_img_resized = fine_img.resize((512, 512), Image.LANCZOS)

    image_tk_1 = ImageTk.PhotoImage(base_img_resized)
    image_tk_2 = ImageTk.PhotoImage(fine_img_resized)

    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_1)
    canvas.create_image(512, 0, anchor=tk.NW, image=image_tk_2)

    canvas.create_text(256, 10, text="Base Model", fill="white", anchor=tk.N)
    canvas.create_text(768, 10, text="Fine-Tuned Model", fill="white", anchor=tk.N)


generate_btn = tk.Button(root, text="Generate Comparison", command=generate)
generate_btn.pack(pady=10)

root.mainloop()
