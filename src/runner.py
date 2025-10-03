import subprocess

files = ["data_script.py", "imbalance_handling.py", "hyperparameter_tuning.py", "model.py"]

for f in files:
    print(f"Running {f}...")
    subprocess.run(["python", f], check=True)