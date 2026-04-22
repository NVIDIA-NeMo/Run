# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end example: run a distributed training job via KubeflowExecutor.

Prerequisites
-------------
* Kubeflow Training Operator v2 installed in your cluster with a
  ``ClusterTrainingRuntime`` named ``"torch-distributed"``.
* A kubeconfig pointing at the target cluster (or run from inside a pod).
* An image that contains your training code and all dependencies.

Usage
-----
    python examples/kubeflow/hello_kubeflow.py \\
        --namespace my-namespace \\
        --image nvcr.io/nvidia/nemo:25.04 \\
        --pvc model-cache

The job runs ``nvidia-smi`` and a quick PyTorch device check on every worker,
then exits.  Swap the ``inline`` script for your real training command.
"""

import argparse
import logging

import nemo_run as run
from nemo_run.core.execution.kubeflow import KubeflowExecutor

logging.basicConfig(level=logging.INFO)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="KubeflowExecutor hello-world example")
parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
parser.add_argument("--image", required=True, help="Container image with your training env")
parser.add_argument("--num-nodes", type=int, default=2, help="Number of worker pods")
parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per pod")
parser.add_argument("--pvc", default=None, help="PVC name for workdir sync (optional)")
parser.add_argument(
    "--runtime-ref",
    default="torch-distributed",
    help="ClusterTrainingRuntime name in your cluster",
)
args = parser.parse_args()

# ── Executor ──────────────────────────────────────────────────────────────────

executor = KubeflowExecutor(
    # Kubeflow TrainJob settings
    launcher=run.Torchrun(),
    runtime_ref=args.runtime_ref,
    namespace=args.namespace,
    image=args.image,
    num_nodes=args.num_nodes,
    gpus_per_node=args.gpus_per_node,

    # Resource requests — tune these to your node type
    cpu_requests="8",
    memory_requests="32Gi",

    # Simple key=value environment variables
    env_vars={
        "NCCL_DEBUG": "INFO",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },

    # Full env var dicts — use for Kubernetes secrets, field references, etc.
    # Example: inject a W&B API key from a Kubernetes Secret named "my-secrets"
    env_list=[
        # {
        #     "name": "WANDB_API_KEY",
        #     "valueFrom": {"secretKeyRef": {"name": "my-secrets", "key": "WANDB_API_KEY"}},
        # },
    ],

    # Toleration that allows scheduling on GPU-tainted nodes
    tolerations=[
        {"effect": "NoSchedule", "key": "nvidia.com/gpu", "operator": "Exists"},
    ],

    # Volumes: a memory-backed /dev/shm so PyTorch DataLoader workers have
    # enough shared memory (the default Kubernetes limit is only 64 MiB).
    volumes=[
        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        *(
            [{"name": "workdir", "persistentVolumeClaim": {"claimName": args.pvc}}]
            if args.pvc
            else []
        ),
    ],
    volume_mounts=[
        {"name": "dshm", "mountPath": "/dev/shm"},
        *(
            [{"name": "workdir", "mountPath": "/nemo-workspace"}]
            if args.pvc
            else []
        ),
    ],

    # Sync the generated launch script to the pod via PVC before launch.
    # Required whenever you use a custom launcher (e.g. run.Torchrun()).
    workdir_pvc=args.pvc,
    workdir_pvc_path="/nemo-workspace",

    labels={"app": "hello-kubeflow"},
)

# ── Task ──────────────────────────────────────────────────────────────────────

# Replace this inline script with your real training command.
script = run.Script(
    inline="""\
nvidia-smi
python - <<'PY'
import os, torch
rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
print(f"rank {rank}/{world} — cuda devices: {torch.cuda.device_count()}")
PY
"""
)

# ── Launch ────────────────────────────────────────────────────────────────────

run.run(script, executor=executor, name="hello-kubeflow")
