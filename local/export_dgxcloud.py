"""
E2E example: Experiment.export() with DGXCloudExecutor

Demonstrates exporting DGX Cloud jobs to a self-contained directory without
any API calls or authentication. The generated script can be inspected and
submitted manually via the DGX Cloud CLI or API.

The output directory contains:
  - <job>_torchrun_job.sh   (the torchrun launch script uploaded to the PVC)
  - submit_all.sh           (calls: bash <job>_torchrun_job.sh for each job)

Run:
    python local/export_dgxcloud.py
    cat /tmp/nemo_export_dgxcloud/train_torchrun_job.sh
"""

import os
import shutil

import nemo_run as run
from nemo_run.core.execution.dgxcloud import DGXCloudExecutor

OUTPUT_DIR = "/tmp/nemo_export_dgxcloud"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def train(model: str, steps: int = 10_000):
    import torch

    print(f"Training {model} on {torch.cuda.device_count()} GPUs for {steps} steps")


# Configure the DGX Cloud executor (credentials are placeholders — not contacted during export)
executor = DGXCloudExecutor(
    base_url="https://api.ngc.nvidia.com/v2/org/my-org/dgxcloud",
    kube_apiserver_url="https://my-cluster.k8s.example.com",
    app_id="my-app-id",
    app_secret="my-app-secret",
    project_name="my-project",
    container_image="nvcr.io/nvidia/nemo:latest",
    pvc_nemo_run_dir="/mnt/pvc/nemo_run",
    pvcs=[{"claimName": "nemo-pvc", "path": "/mnt/pvc"}],
    nodes=2,
    gpus_per_node=8,
    packager=run.GitArchivePackager(),
)

with run.Experiment("export-dgxcloud-demo") as exp:
    exp.add(
        run.Partial(train, model="mistral-7b", steps=100_000),
        executor=executor,
        name="train",
    )
    exp.export(OUTPUT_DIR)

files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nExported files: {files}")

torchrun_scripts = [f for f in files if f.endswith("_torchrun_job.sh")]
assert len(torchrun_scripts) == 1, f"Expected 1 torchrun script, got: {torchrun_scripts}"

print("\n--- train_torchrun_job.sh (first 40 lines) ---")
with open(f"{OUTPUT_DIR}/{torchrun_scripts[0]}") as f:
    lines = f.readlines()
    print("".join(lines[:40]))

print("--- submit_all.sh ---")
with open(f"{OUTPUT_DIR}/submit_all.sh") as f:
    print(f.read())
