import os
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

class Args:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"
    output_dir = "lora_output"
    resolution = 512
    base_model_cache_dir = "base_model_cache"

args = Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model_name = "stabilityai/stable-diffusion-3.5-medium"
os.makedirs(args.base_model_cache_dir, exist_ok=True)

print("Loading base model...")
base_generator = DiffusionPipeline.from_pretrained(
    base_model_name,
    cache_dir=args.base_model_cache_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16"
).to(device)
print("Base model loaded.")

print("Loading fine-tuned model...")
fine_tuned_pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16"
).to(device)

if "transformer" not in fine_tuned_pipe.components:
    raise AttributeError("Fine-tuned pipeline does not include a 'transformer'.")

print("Applying LoRA weights...")
transformer = fine_tuned_pipe.components["transformer"]

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

root = tk.Tk()
root.title("Text-to-Tile Generator - Base vs Fine-Tuned")
root.geometry("1080x900")

prompt_label = tk.Label(root, text="Enter Tile Description:")
prompt_label.pack(pady=5)

prompt_entry = tk.Entry(root, width=60)
prompt_entry.pack(pady=5)
prompt_entry.insert(0, "A top-down view of a snow tile, pixel art, top-down")

res_label = tk.Label(root, text="Choose Resolution:")
res_label.pack(pady=5)

resolution_var = tk.StringVar(value="128")
res_dropdown = ttk.Combobox(root, textvariable=resolution_var, state="readonly")
res_dropdown["values"] = ["16", "32", "64", "128", "512"]
res_dropdown.pack(pady=5)

canvas = tk.Canvas(root, width=1024, height=512)
canvas.pack(pady=10)

image_tk_1 = image_tk_2 = None
base_img_saved = fine_img_saved = None

def generate():
    global image_tk_1, image_tk_2, base_img_saved, fine_img_saved

    prompt = prompt_entry.get()
    res = int(resolution_var.get())

    if not prompt:
        messagebox.showerror("Error", "Please enter a description.")
        return

    with torch.autocast(device_type=device.type):
        base_img = base_generator(prompt, height=res, width=res, num_inference_steps=30).images[0]
        fine_img = fine_tuned_pipe(prompt, height=res, width=res, num_inference_steps=30).images[0]

    base_img_resized = base_img.resize((512, 512), Image.LANCZOS)
    fine_img_resized = fine_img.resize((512, 512), Image.LANCZOS)

    base_img_saved = base_img
    fine_img_saved = fine_img

    image_tk_1 = ImageTk.PhotoImage(base_img_resized)
    image_tk_2 = ImageTk.PhotoImage(fine_img_resized)

    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_1)
    canvas.create_image(512, 0, anchor=tk.NW, image=image_tk_2)
    canvas.create_text(256, 10, text="Base Model", fill="white", anchor=tk.N)
    canvas.create_text(768, 10, text="Fine-Tuned Model", fill="white", anchor=tk.N)
    print("Finished generating...")

def save_base():
    if base_img_saved:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            base_img_saved.save(path)
            messagebox.showinfo("Saved", f"Base model image saved to:\n{path}")

def save_finetuned():
    if fine_img_saved:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            fine_img_saved.save(path)
            messagebox.showinfo("Saved", f"Fine-tuned image saved to:\n{path}")

generate_btn = tk.Button(root, text="Generate Comparison", command=generate)
generate_btn.pack(pady=10)

save_base_btn = tk.Button(root, text="Save Base Image", command=save_base)
save_base_btn.pack(pady=5)

save_finetuned_btn = tk.Button(root, text="Save Fine-Tuned Image", command=save_finetuned)
save_finetuned_btn.pack(pady=5)

root.mainloop()
