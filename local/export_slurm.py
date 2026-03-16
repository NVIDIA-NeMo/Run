"""
E2E example: Experiment.export() with SlurmExecutor

Demonstrates exporting SLURM jobs to a self-contained directory without
connecting to any cluster or submitting any job.

The output directory contains:
  - <job>_sbatch.sh   (ready-to-submit sbatch script)
  - submit_all.sh     (calls: sbatch <job>_sbatch.sh for each job)

To actually submit after export (requires cluster access):
    sbatch /tmp/nemo_export_slurm/pretrain_sbatch.sh

Run:
    python local/export_slurm.py
    cat /tmp/nemo_export_slurm/pretrain_sbatch.sh
"""

import os
import shutil

import nemo_run as run
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.tunnel.client import SSHTunnel

OUTPUT_DIR = "/tmp/nemo_export_slurm"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def pretrain(model: str, num_steps: int = 1000, lr: float = 3e-4):
    print(f"Pre-training {model} for {num_steps} steps at lr={lr}")


def finetune(model: str, dataset: str, epochs: int = 5):
    print(f"Fine-tuning {model} on {dataset} for {epochs} epochs")


# Configure the SLURM executor (no real cluster needed for export)
def make_slurm_executor(nodes: int = 1) -> SlurmExecutor:
    tunnel = SSHTunnel(
        host="my-cluster.example.com",  # placeholder — not contacted during export
        user="myuser",
        job_dir="/scratch/myuser/nemo_jobs",
    )
    return SlurmExecutor(
        account="my_account",
        partition="gpu",
        nodes=nodes,
        ntasks_per_node=8,
        gpus_per_node=8,
        container_image="nvcr.io/nvidia/nemo:latest",
        time="04:00:00",
        tunnel=tunnel,
        packager=run.GitArchivePackager(),
    )


with run.Experiment("export-slurm-demo") as exp:
    exp.add(
        run.Partial(pretrain, model="llama-7b", num_steps=50_000, lr=1e-4),
        executor=make_slurm_executor(nodes=4),
        name="pretrain",
    )
    exp.add(
        run.Partial(finetune, model="llama-7b", dataset="squad", epochs=3),
        executor=make_slurm_executor(nodes=1),
        name="finetune",
    )
    exp.export(OUTPUT_DIR)

files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nExported files: {files}")

sbatch_scripts = [f for f in files if f.endswith("_sbatch.sh")]
assert len(sbatch_scripts) == 2, f"Expected 2 sbatch scripts, got: {sbatch_scripts}"

print("\n--- pretrain_sbatch.sh (first 40 lines) ---")
with open(f"{OUTPUT_DIR}/pretrain_sbatch.sh") as f:
    lines = f.readlines()
    print("".join(lines[:40]))

print("--- submit_all.sh ---")
with open(f"{OUTPUT_DIR}/submit_all.sh") as f:
    print(f.read())
