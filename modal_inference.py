import modal
import os
import subprocess
import sys

# Define the image
# Copies local dir "." to "/root/bdh" in the container
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    # Added 'pathway' to requirements, so handled there.
    # IMPORTANT: Ensure the inference/ package is fully uploaded
    .add_local_dir(".", "/root/bdh", ignore=["venv", ".venv", "env", "__pycache__", "output", ".git"])
)

app = modal.App("bdh-inference")

@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=32768,
    timeout=14400,  # 4 hours
)
def run_inference(mode: str = "batch"):
    """
    Runs the BDH inference script on the allocated GPU.
    
    Args:
        mode: "batch" or "stream" (Pathway mode)
    """
    # Change directory to the copied repo root inside the container
    repo_root = "/root/bdh"
    os.chdir(repo_root)
    
    # Verify required files exist
    required_files = [
        "bdh.py",
        "inference.py",
        "inference/__init__.py",  # Check modular package exists
        "files/backstory.txt",
        "files/novel.txt"
    ]
    
    print(f"Current working directory: {os.getcwd()}")
    print("Verifying files...")
    
    for f in required_files:
        if not os.path.exists(f):
            # Try to be helpful if files are in different location or named differently
            if f.startswith("files/"):
                print(f"  ? Warning: {f} not found (using defaults?)")
                continue
            raise RuntimeError(f"Required file not found: {f}")
        print(f"  ✓ Found {f}")
        
    print(f"Starting inference process in {mode.upper()} mode...")
    sys.stdout.flush()
    
    # Run the inference script
    # streaming stdout/stderr to the container logs
    cmd = [sys.executable, "-u", "inference.py"]
    if mode == "stream":
        cmd.extend(["--mode", "stream"])
        
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
    except subprocess.CalledProcessError as e:
        # Raise error if inference script fails (non-zero exit code)
        raise RuntimeError(f"Inference script failed with exit code {e.returncode}") from e
    
    print("Inference completed successfully.")

@app.local_entrypoint()
def main(mode: str = "batch"):
    """
    Run the inference function on Modal.
    Usage: modal run modal_inference.py --mode stream
    """
    run_inference.remote(mode=mode)
