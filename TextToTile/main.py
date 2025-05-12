
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
import uuid
import os
import tkinter as tk
from tkinter import messagebox, filedialog

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading generative model...")
generator = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    revision="fp16" if device == "cuda" else "main",
    use_auth_token=True
).to(device)
print("Model loaded successfully.")

os.makedirs("generated_tiles", exist_ok=True)

root = tk.Tk()
root.title("Text-to-Tile Generator")
root.geometry("600x800")

prompt_label = tk.Label(root, text="Enter Tile Description:")
prompt_label.pack(pady=10)

prompt_entry = tk.Entry(root, width=50)
prompt_entry.pack(pady=5)

number_label = tk.Label(root, text="Number of Tiles to Generate:")
number_label.pack(pady=10)

number_entry = tk.Entry(root, width=10)
number_entry.pack(pady=5)
number_entry.insert(0, "1")

image_label = tk.Label(root)
image_label.pack(pady=10)

current_grid_image = None

basic_colors = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "brown": (165, 42, 42),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255)
}

accepted_matches = {
    "blue": ["blue", "cyan"],
    "green": ["green"],
    "yellow": ["yellow"],
    "white": ["white"],
    "gray": ["gray", "silver"],
    "red": ["red", "brown"],
    "brown": ["brown", "red"],
    "black": ["black"],
    "orange": ["orange"],
    "purple": ["purple", "magenta"],
}

def closest_color(requested_color):
    min_dist = float('inf')
    closest_name = None
    for name, rgb in basic_colors.items():
        rd = (rgb[0] - requested_color[0]) ** 2
        gd = (rgb[1] - requested_color[1]) ** 2
        bd = (rgb[2] - requested_color[2]) ** 2
        distance = rd + gd + bd
        if distance < min_dist:
            min_dist = distance
            closest_name = name
    return closest_name

def get_dominant_color(image):
    image = image.resize((32, 32))
    pixels = image.getcolors(32 * 32)
    most_common_pixel = max(pixels, key=lambda t: t[0])[1]
    return closest_color(most_common_pixel)

def infer_expected_color(description):
    description = description.lower()
    if "forest" in description or "tree" in description:
        return "green"
    if "desert" in description or "sand" in description:
        return "yellow"
    if "snow" in description or "ice" in description:
        return "white"
    if "water" in description or "lake" in description or "river" in description:
        return "blue"
    if "mountain" in description or "rock" in description:
        return "gray"
    if "volcano" in description or "lava" in description:
        return "red"
    return None

def is_black_image(image):
    extrema = image.getextrema()
    return all(channel == (0, 0) for channel in extrema)

def generate_tiles():
    global current_grid_image

    description = prompt_entry.get()
    try:
        number_of_tiles = int(number_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number.")
        return

    if not description:
        messagebox.showerror("Error", "Please enter a description.")
        return

    expected_color = infer_expected_color(description)
    first_tile = None

    for i in range(number_of_tiles):
        success = False
        attempts = 0
        max_attempts = 5

        while not success and attempts < max_attempts:
            full_prompt = f"{description}, detailed pixel art, RPG style, colorful, seamless texture, top-down view, natural colors, realistic lighting"

            try:
                with torch.autocast(device_type=device):
                    image = generator(full_prompt, height=512, width=512, num_inference_steps=30).images[0]
                image = image.resize((64, 64), Image.LANCZOS)

                if is_black_image(image):
                    attempts += 1
                    continue

                filename = f"generated_tiles/{description}_{uuid.uuid4().hex}.png"
                image.save(filename)

                dominant_color = get_dominant_color(image)

                if expected_color:
                    valid_colors = accepted_matches.get(expected_color, [expected_color])
                    if any(valid in dominant_color for valid in valid_colors):
                        success = True
                        print(f"Tile generated and matches expected color ({expected_color}).")
                        if i == 0:
                            first_tile = image
                    else:
                        print(f"Color mismatch (Expected: {expected_color}, Got: {dominant_color}). Retrying...")
                        attempts += 1
                else:
                    success = True
                    if i == 0:
                        first_tile = image

            except Exception as e:
                messagebox.showerror("Error", f"Generation failed: {e}")
                return

        if not success:
            messagebox.showwarning("Warning", f"Could not match color after {max_attempts} attempts for tile {i+1}.")

    if first_tile:
        grid_size = 5
        grid_image = Image.new('RGB', (64 * grid_size, 64 * grid_size))
        for y in range(grid_size):
            for x in range(grid_size):
                grid_image.paste(first_tile, (x * 64, y * 64))

        img = ImageTk.PhotoImage(grid_image)
        image_label.configure(image=img)
        image_label.image = img

        current_grid_image = grid_image

def save_preview():
    if current_grid_image:
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filepath:
            current_grid_image.save(filepath)
            messagebox.showinfo("Saved", f"Preview grid saved to {filepath}")
    else:
        messagebox.showerror("Error", "No preview available to save.")

generate_button = tk.Button(root, text="Generate Tiles", command=generate_tiles)
generate_button.pack(pady=10)

save_button = tk.Button(root, text="Save Preview Grid", command=save_preview)
save_button.pack(pady=10)

root.mainloop()
