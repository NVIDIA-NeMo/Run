"""
E2E example: Experiment.export() with LocalExecutor

Demonstrates exporting a single job to a self-contained script directory
without submitting anything. The output directory contains:
  - hello-job.sh    (executable bash script)
  - submit_all.sh   (launcher that calls: bash hello-job.sh)

Run:
    python local/export_local.py
    ls /tmp/nemo_export_local/
    cat /tmp/nemo_export_local/hello-job.sh
    bash /tmp/nemo_export_local/submit_all.sh
"""

import os
import shutil

import nemo_run as run
from nemo_run.core.execution.local import LocalExecutor

OUTPUT_DIR = "/tmp/nemo_export_local"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def greet(name: str, times: int = 1):
    for _ in range(times):
        print(f"Hello, {name}!")


with run.Experiment("export-local-demo") as exp:
    task = run.Partial(greet, name="NeMo", times=3)
    exp.add(task, executor=LocalExecutor(), name="hello-job")
    exp.export(OUTPUT_DIR)

files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nExported files: {files}")
assert "hello-job.sh" in files, "Expected hello-job.sh"
assert "submit_all.sh" in files, "Expected submit_all.sh"

print("\n--- hello-job.sh ---")
with open(f"{OUTPUT_DIR}/hello-job.sh") as f:
    print(f.read())

print("--- submit_all.sh ---")
with open(f"{OUTPUT_DIR}/submit_all.sh") as f:
    print(f.read())
