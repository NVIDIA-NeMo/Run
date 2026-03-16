"""
E2E example: Experiment.export() with run.Script tasks

Shows that export() works with shell Script tasks (not just Partial),
which is a common pattern for SLURM-style jobs where the user provides
a raw bash script.

The exported .sh file wraps the inline command; submit_all.sh calls
`bash <script>.sh` for each job.

Run:
    python local/export_script.py
    bash /tmp/nemo_export_script/submit_all.sh
"""

import os
import shutil

import nemo_run as run
from nemo_run.core.execution.local import LocalExecutor

OUTPUT_DIR = "/tmp/nemo_export_script"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

with run.Experiment("export-script-demo") as exp:
    exp.add(
        run.Script(inline="echo 'Starting data download'; sleep 1; echo 'Done'"),
        executor=LocalExecutor(),
        name="download",
    )
    exp.add(
        run.Script(inline="echo 'Unpacking archive'; sleep 1; echo 'Unpacked'"),
        executor=LocalExecutor(),
        name="unpack",
    )
    exp.export(OUTPUT_DIR)

files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nExported files: {files}")
print("\n--- submit_all.sh ---")
with open(f"{OUTPUT_DIR}/submit_all.sh") as f:
    print(f.read())
