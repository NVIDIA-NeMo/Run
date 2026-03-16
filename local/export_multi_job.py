"""
E2E example: Experiment.export() with multiple LocalExecutor jobs

Demonstrates exporting multiple jobs to a shared output directory.
Each job produces its own .sh script; submit_all.sh chains them all.

Run:
    python local/export_multi_job.py
    ls /tmp/nemo_export_multi/
    bash /tmp/nemo_export_multi/submit_all.sh
"""

import os
import shutil

import nemo_run as run
from nemo_run.core.execution.local import LocalExecutor

OUTPUT_DIR = "/tmp/nemo_export_multi"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def preprocess(dataset: str, workers: int = 4):
    print(f"Preprocessing {dataset} with {workers} workers")


def train(model: str, epochs: int = 10, lr: float = 1e-3):
    print(f"Training {model} for {epochs} epochs at lr={lr}")


def evaluate(model: str, split: str = "test"):
    print(f"Evaluating {model} on {split} split")


with run.Experiment("export-multi-demo") as exp:
    exp.add(
        run.Partial(preprocess, dataset="imagenet", workers=8),
        executor=LocalExecutor(),
        name="preprocess",
    )
    exp.add(
        run.Partial(train, model="resnet50", epochs=50, lr=5e-4),
        executor=LocalExecutor(),
        name="train",
    )
    exp.add(
        run.Partial(evaluate, model="resnet50", split="val"),
        executor=LocalExecutor(),
        name="evaluate",
    )
    exp.export(OUTPUT_DIR)

files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nExported files: {files}")

sh_scripts = [f for f in files if f.endswith(".sh") and f != "submit_all.sh"]
assert len(sh_scripts) == 3, f"Expected 3 job scripts, got: {sh_scripts}"

print("\n--- submit_all.sh ---")
with open(f"{OUTPUT_DIR}/submit_all.sh") as f:
    print(f.read())
