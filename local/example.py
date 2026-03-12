import time

from nemo_run.core.execution.pytorchjob import PyTorchJobExecutor

EXPECTED_LOG_CONTENT = "NEMO_TEST_OK"

e = PyTorchJobExecutor(
    namespace="runai-nemo-ci",
    image="nvcr.io/nvidian/nemo:nightly",
    num_workers=2,
    nproc_per_node=8,
    gpus_per_node=8,
    cpu_requests="16",
    memory_requests="64Gi",
    volumes=[
        {
            "name": "model-cache",
            "persistentVolumeClaim": {"claimName": "nemo-ci-datasets-project-nkf5l"},
        }
    ],
    volume_mounts=[{"name": "model-cache", "mountPath": "/nemo-workspace"}],
    labels={"app": "nemo-ci-training"},
)

# Script: print the sentinel, then sleep so we can read logs and cancel cleanly
cmd = [
    "/bin/bash",
    "-c",
    f"echo 'print(\"{EXPECTED_LOG_CONTENT}\"); import time; time.sleep(300)' > /tmp/test.py && "
    "torchrun --nnodes=$PET_NNODES --nproc_per_node=$PET_NPROC_PER_NODE "
    "--node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT /tmp/test.py",
]

# ── Launch and wait until RUNNING ────────────────────────────────────────────
job_name, state = e.launch("nemo-ci-training", cmd, wait=True, timeout=300)
print(f"Launched: {job_name}, state: {state}")

# ── Fetch logs and verify sentinel ────────────────────────────────────────────
print("Polling logs until sentinel appears (up to 2 min)...")
logs = []
deadline = time.time() + 120
while time.time() < deadline:
    logs = list(e.fetch_logs(job_name, stream=False, lines=50))
    if any(EXPECTED_LOG_CONTENT in line for line in logs):
        break
    print(f"  waiting for sentinel ({len(logs)} lines so far)...")
    time.sleep(5)

print(f"  received {len(logs)} lines")
for line in logs[:5]:
    print(f"  | {line}")

assert any(EXPECTED_LOG_CONTENT in line for line in logs), (
    f"Expected '{EXPECTED_LOG_CONTENT}' not found in logs.\nGot: {logs}"
)
print(f"✓ Log sentinel '{EXPECTED_LOG_CONTENT}' verified")

# ── Cancel and wait for full cleanup ─────────────────────────────────────────
print("Cancelling job and waiting for cleanup...")
cleaned = e.cancel(job_name, wait=True, timeout=120)
assert cleaned, "Cleanup failed — pods or CR still present after timeout"
print("Full cycle complete without kubectl")
