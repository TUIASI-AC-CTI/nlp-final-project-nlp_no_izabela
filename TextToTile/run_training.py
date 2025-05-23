import subprocess
import os

# Cale spre python-ul din mediul virtual
python_path = os.path.join(os.environ["VIRTUAL_ENV"], "Scripts", "python.exe")

# Argumente pentru scriptul de antrenare (fără --validation_prompt și --report_to)
cmd = [
    python_path,
    "train_lora.py",
    "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3.5-medium",
    "--train_data_dir=data/dataset_lora",
    "--output_dir=lora_output",
    "--resolution=512",
    "--train_batch_size=1",
    "--num_train_epochs=10",
    "--learning_rate=1e-4",
    "--checkpointing_steps=500"
]

# Rularea scriptului
subprocess.run(cmd)
