import subprocess
import os

python_path = os.path.join(os.environ["VIRTUAL_ENV"], "Scripts", "python.exe")

cmd = [
    python_path,
    "train_lora.py",
    "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
    "--train_data_dir=data/dataset_lora",
    "--output_dir=lora_output",
    "--resolution=512",
    "--train_batch_size=1",
    "--num_train_epochs=50",
    "--learning_rate=1e-4",
    "--checkpointing_steps=500"
]

subprocess.run(cmd)
