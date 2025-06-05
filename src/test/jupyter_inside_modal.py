# ---
# args: ["--timeout", 10]
# ---

# ## Overview
#
# Quick snippet showing how to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal Volumes.
# This uses [Modal Tunnels](https://modal.com/docs/guide/tunnels#tunnels-beta)
# to create a tunnel between the running Jupyter instance and the internet.
#
# If you want to your Jupyter notebook to run _locally_ and execute remote Modal Functions in certain cells, see the `basic.ipynb` example :)

import os
import subprocess
import time

import modal

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "opencv-contrib-python",
        "huggingface-hub",
        "Pillow",
        "numpy",
        "jupyter",
        # "opencv-contrib-python-headless",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("git+https://github.com/luca-medeiros/lang-segment-anything.git")
)
app = modal.App("ImageAlfread_Jupyter", image=image)

volume = modal.Volume.from_name(
    "ImageAlfred-data", create_if_missing=True
)

CACHE_DIR = "/root/cache"
JUPYTER_TOKEN = "GCL!ph9IjX4Apk9" 


@app.function(volumes={CACHE_DIR: volume})
def seed_volume():
    pass


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.


@app.function(max_containers=1, volumes={CACHE_DIR: volume}, timeout=1_500)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 10_000):
    # Write some images to a volume, for demonstration purposes.
    seed_volume.remote()
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)


# Doing `modal run jupyter_inside_modal.py` will run a Modal app which starts
# the Juypter server at an address like https://u35iiiyqp5klbs.r3.modal.host.
# Visit this address in your browser, and enter the security token
# you set for `JUPYTER_TOKEN`.
