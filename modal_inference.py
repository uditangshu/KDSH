import modal
import os
import subprocess
import sys

# Define the image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("bdh-inference")

@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=32768,
    timeout=14400,  # 4 hours
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/bdh")]
)
def run_inference():
    """
    Runs the BDH inference script on the allocated GPU.
    """
    # Change directory to the repo root inside the container
    repo_root = "/root/bdh"
    os.chdir(repo_root)
    
    # Verify required files exist
    required_files = [
        "inference.py",
        "files/backstory.txt",
        "files/novel.txt"
    ]
    
    print(f"Current working directory: {os.getcwd()}")
    print("Verifying files...")
    
    for f in required_files:
        if not os.path.exists(f):
            raise RuntimeError(f"Required file not found: {f}")
        print(f"  ✓ Found {f}")
        
    print("Starting inference process...")
    sys.stdout.flush()
    
    # Run the inference script
    # streaming stdout/stderr to the container logs
    try:
        subprocess.run(
            [sys.executable, "inference.py"],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
    except subprocess.CalledProcessError as e:
        # Raise error if inference script fails (non-zero exit code)
        raise RuntimeError(f"Inference script failed with exit code {e.returncode}") from e
    
    print("Inference completed successfully.")

if __name__ == "__main__":
    app.serve()
