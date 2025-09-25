---
description: "Integration guides for connecting NeMo Run with cloud platforms like AWS, GCP, Azure, and other cloud providers."
categories: ["integrations-apis"]
tags: ["integrations", "cloud-platforms", "aws", "gcp", "azure", "kubernetes", "docker"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "text-only"
---

(cloud-platforms)=

# Cloud Platforms Integration

This guide covers integrating NeMo Run with popular cloud platforms to scale your ML experiments across distributed computing resources.

(cloud-supported-platforms)=
## Supported Cloud Platforms

NeMo Run supports integration with major cloud providers:

- **AWS (Amazon Web Services)** - EC2, EKS, SageMaker
- **Google Cloud Platform (GCP)** - Compute Engine, GKE, Vertex AI
- **Microsoft Azure** - Virtual Machines, AKS, Machine Learning
- **NVIDIA DGX Cloud** - DGX Cloud clusters
- **Kubernetes** - Any Kubernetes cluster
- **Docker** - Containerized execution

(cloud-aws)=
## AWS Integration

Run on EC2 and EKS using DockerExecutor-based setups. Supply credentials via environment variables and choose GPU-enabled instances as needed.

(cloud-aws-ec2)=
### EC2 Instance Configuration

```python
import nemo_run as run

# EC2 instance configuration
ec2_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key",
        "AWS_DEFAULT_REGION": "us-west-2"
    }
)

# Training function
def train_on_ec2(model_config, dataset):
    model = model_config.build()
    # Training logic
    return model

# Run experiment on EC2
with run.Experiment("ec2_training") as experiment:
    experiment.add(
        run.Partial(train_on_ec2, model_config, dataset),
        name="ec2_training",
        executor=ec2_executor
    )
    experiment.run()
```

### EKS (Elastic Kubernetes Service) Integration

```python
import nemo_run as run

# EKS cluster configuration using Docker executor
eks_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "KUBECONFIG": "/path/to/kubeconfig",
        "AWS_DEFAULT_REGION": "us-west-2"
    }
)

# GPU-enabled pod configuration
gpu_pod_config = {
    "node_selector": {
        "node.kubernetes.io/instance-type": "g4dn.xlarge"
    },
    "tolerations": [
        {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule"
        }
    ]
}

# Run experiment on EKS
run.run(
    run.Partial(train_on_ec2, model_config, dataset),
    executor=eks_executor
)
```

### Spot Instance Configuration

```python
import nemo_run as run

# Spot instance configuration for cost optimization
spot_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "SPOT_INSTANCE": "true",
        "MAX_BID_PRICE": "0.50"
    }
)

# Fault-tolerant training
def fault_tolerant_training(model_config, dataset, checkpoint_path: str):
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        model = load_checkpoint(model_config, checkpoint_path)
    else:
        model = model_config.build()

    # Training with periodic checkpointing
    for epoch in range(epochs):
        train_epoch(model, dataset)
        save_checkpoint(model, f"{checkpoint_path}_epoch_{epoch}")

    return model

# Run with spot instances
run.run(
    run.Partial(fault_tolerant_training, model_config, dataset, "/tmp/checkpoint"),
    executor=spot_executor
)
```

(cloud-gcp)=
## Google Cloud Platform (GCP) Integration

Use Compute Engine and GKE to execute NeMo Run experiments, authenticating via service accounts and environment variables.

(cloud-gcp-compute)=
### Compute Engine Configuration

```python
import nemo_run as run

# GCP Compute Engine configuration
gcp_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json",
        "GCP_PROJECT_ID": "your-project-id",
        "GCP_ZONE": "us-central1-a"
    }
)

# Training function
def train_on_gcp(model_config, dataset):
    model = model_config.build()
    # Training logic
    return model

# Run experiment on GCP
with run.Experiment("gcp_training") as experiment:
    experiment.add(
        run.Partial(train_on_gcp, model_config, dataset),
        name="gcp_training",
        executor=gcp_executor
    )
    experiment.run()
```

### GKE (Google Kubernetes Engine) Integration

```python
import nemo_run as run

# GKE cluster configuration
gke_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "GKE_CLUSTER_NAME": "nemo-run-cluster",
        "GKE_ZONE": "us-central1-a"
    }
)

# Run experiment on GKE
run.run(
    run.Partial(train_on_gcp, model_config, dataset),
    executor=gke_executor
)
```

(cloud-azure)=
## Microsoft Azure Integration

Target Azure VMs and AKS clusters with DockerExecutor, passing Azure credentials through environment variables.

(cloud-azure-vm)=
### Azure Virtual Machines

```python
import nemo_run as run

# Azure VM configuration
azure_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "AZURE_CLIENT_ID": "your-client-id",
        "AZURE_CLIENT_SECRET": "your-client-secret",
        "AZURE_TENANT_ID": "your-tenant-id",
        "AZURE_SUBSCRIPTION_ID": "your-subscription-id"
    }
)

# Training function
def train_on_azure(model_config, dataset):
    model = model_config.build()
    # Training logic
    return model

# Run experiment on Azure
with run.Experiment("azure_training") as experiment:
    experiment.add(
        run.Partial(train_on_azure, model_config, dataset),
        name="azure_training",
        executor=azure_executor
    )
    experiment.run()
```

(cloud-azure-aks)=
### AKS (Azure Kubernetes Service) Integration

```python
import nemo_run as run

# AKS cluster configuration
aks_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "AKS_CLUSTER_NAME": "nemo-run-cluster",
        "AKS_RESOURCE_GROUP": "your-resource-group"
    }
)

# Run experiment on AKS
run.run(
    run.Partial(train_on_azure, model_config, dataset),
    executor=aks_executor
)
```

(cloud-dgx)=
## NVIDIA DGX Cloud Integration

Submit experiments to DGX Cloud clusters using the dedicated executor and your project and cluster identifiers.

(cloud-dgx-config)=
### DGX Cloud Configuration

```python
import nemo_run as run

# DGX Cloud executor configuration
dgx_executor = run.DGXCloudExecutor(
    project_id="your-project-id",
    cluster_id="your-cluster-id",
    node_count=1,
    gpus_per_node=8,
    image="nvidia/pytorch:24.05-py3"
)

# Training function
def train_on_dgx(model_config, dataset):
    model = model_config.build()
    # Training logic
    return model

# Run experiment on DGX Cloud
with run.Experiment("dgx_training") as experiment:
    experiment.add(
        run.Partial(train_on_dgx, model_config, dataset),
        name="dgx_training",
        executor=dgx_executor
    )
    experiment.run()
```

(cloud-k8s)=
## Kubernetes Integration

Run against generic Kubernetes clusters by pointing DockerExecutor at a kubeconfig and GPU-enabled nodes.

(cloud-k8s-generic)=
### Generic Kubernetes Configuration

```python
import nemo_run as run

# Generic Kubernetes executor
k8s_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    env_vars={
        "KUBECONFIG": "/path/to/kubeconfig"
    }
)

# Training function
def train_on_k8s(model_config, dataset):
    model = model_config.build()
    # Training logic
    return model

# Run experiment on Kubernetes
run.run(
    run.Partial(train_on_k8s, model_config, dataset),
    executor=k8s_executor
)
```

(cloud-docker)=
## Docker Integration

Containerize local and remote runs with DockerExecutor, mounting volumes and selecting GPU resources.

(cloud-docker-local)=
### Local Docker Execution

```python
import nemo_run as run

# Local Docker executor
docker_executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    volumes=[
        "/host/data:/container/data",
        "/host/models:/container/models"
    ]
)

# Training function
def train_in_docker(model_config, dataset):
    model = model_config.build()
    # Training logic
    return model

# Run experiment in Docker
with run.Experiment("docker_training") as experiment:
    experiment.add(
        run.Partial(train_in_docker, model_config, dataset),
        name="docker_training",
        executor=docker_executor
    )
    experiment.run()
```

(cloud-multicloud)=
## Multi-Cloud Configuration

Select executors dynamically based on the active cloud provider to reuse the same experiment code anywhere.

(cloud-multicloud-envaware)=
### Environment-Aware Configuration

```python
import nemo_run as run
import os
from typing import Dict, Any

class CloudConfig:
    def __init__(self, cloud_provider: str):
        self.cloud_provider = cloud_provider
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration based on cloud provider."""
        if self.cloud_provider == "aws":
            return {
                "executor": run.DockerExecutor(
                    container_image="nvidia/pytorch:24.05-py3",
                    num_gpus=1,
                    env_vars={
                        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
                        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY")
                    }
                )
            }
        elif self.cloud_provider == "gcp":
            return {
                "executor": run.DockerExecutor(
                    container_image="nvidia/pytorch:24.05-py3",
                    num_gpus=1,
                    env_vars={
                        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                    }
                )
            }
        elif self.cloud_provider == "azure":
            return {
                "executor": run.DockerExecutor(
                    container_image="nvidia/pytorch:24.05-py3",
                    num_gpus=1,
                    env_vars={
                        "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID"),
                        "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET")
                    }
                )
            }
        else:
            return {
                "executor": run.LocalExecutor()
            }

    def get_executor(self):
        return self.config["executor"]

# Usage
aws_config = CloudConfig("aws")
gcp_config = CloudConfig("gcp")

# Run on AWS
with run.Experiment("aws_training") as experiment:
    experiment.add(
        run.Partial(train_model, model_config),
        name="aws_training",
        executor=aws_config.get_executor()
    )
    experiment.run()

# Run on GCP
with run.Experiment("gcp_training") as experiment:
    experiment.add(
        run.Partial(train_model, model_config),
        name="gcp_training",
        executor=gcp_config.get_executor()
    )
    experiment.run()
```

(cloud-cost)=
## Cost Optimization Strategies

Use spot instances and auto-scaling to reduce costs while maintaining throughput and reliability.

(cloud-cost-spot)=
### Spot Instance Management

```python
import nemo_run as run

def create_cost_optimized_executor(cloud_provider: str, use_spot: bool = True):
    """Create cost-optimized executor with spot instances."""

    base_config = {
        "container_image": "nvidia/pytorch:24.05-py3",
        "num_gpus": 1
    }

    if use_spot:
        base_config.setdefault("env_vars", {})
        base_config["env_vars"].update({
            "SPOT_INSTANCE": "true",
            "MAX_BID_PRICE": "0.50"
        })

    if cloud_provider == "aws":
        base_config.setdefault("env_vars", {})
        base_config["env_vars"].update({
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY")
        })

    return run.DockerExecutor(**base_config)

# Usage
cost_optimized_executor = create_cost_optimized_executor("aws", use_spot=True)

run.run(
    run.Partial(train_model, model_config),
    executor=cost_optimized_executor
)
```

### Auto-scaling Configuration

```python
import nemo_run as run

def create_auto_scaling_config(cloud_provider: str):
    """Create auto-scaling configuration."""

    base_config = {
        "container_image": "nvidia/pytorch:24.05-py3",
        "num_gpus": 1
    }

    if cloud_provider == "aws":
        base_config.setdefault("env_vars", {})
        base_config["env_vars"].update({
            "AUTO_SCALING": "true",
            "MIN_NODES": "1",
            "MAX_NODES": "10",
            "TARGET_CPU_UTILIZATION": "70"
        })

    return run.DockerExecutor(**base_config)

# Usage
auto_scaling_executor = create_auto_scaling_config("aws")

run.run(
    run.Partial(train_model, model_config),
    executor=auto_scaling_executor
)
```

(cloud-security)=
## Security Best Practices

Manage credentials securely with environment variables and least-privilege access across providers.

(cloud-security-credentials)=
### Credential Management

```python
import nemo_run as run
import os
from typing import Dict, Any

def create_secure_executor(cloud_provider: str) -> run.Executor:
    """Create executor with secure credential management."""

    # Load credentials from environment variables
    credentials = {
        "aws": {
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN")
        },
        "gcp": {
            "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        },
        "azure": {
            "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID"),
            "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET"),
            "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID")
        }
    }

    base_config = {
        "container_image": "nvidia/pytorch:24.05-py3",
        "num_gpus": 1,
        "env_vars": credentials.get(cloud_provider, {})
    }

    return run.DockerExecutor(**base_config)

# Usage
secure_executor = create_secure_executor("aws")

run.run(
    run.Partial(train_model, model_config),
    executor=secure_executor
)
```

(cloud-monitoring)=
## Monitoring and Logging

Add provider-specific monitoring to your runs and capture logs and metrics alongside experiments.

(cloud-monitoring-specific)=
### Cloud-Specific Monitoring

```python
import nemo_run as run
import logging

def setup_cloud_monitoring(cloud_provider: str):
    """Setup cloud-specific monitoring and logging."""

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Cloud-specific monitoring setup
    if cloud_provider == "aws":
        logger.info("Setting up AWS CloudWatch monitoring")
        # AWS CloudWatch configuration
    elif cloud_provider == "gcp":
        logger.info("Setting up GCP Stackdriver monitoring")
        # GCP Stackdriver configuration
    elif cloud_provider == "azure":
        logger.info("Setting up Azure Monitor")
        # Azure Monitor configuration

    return logger

# Usage
logger = setup_cloud_monitoring("aws")

def train_with_monitoring(model_config, dataset):
    logger.info("Starting training on cloud")
    model = model_config.build()
    # Training logic
    logger.info("Training completed")
    return model

# Run with monitoring
run.run(
    run.Partial(train_with_monitoring, model_config, dataset),
    executor=secure_executor
)
```

(cloud-next-steps)=
## Next Steps

Explore related integration guides to extend your cloud workflows with NeMo Run.

- Explore [ML Frameworks Integration](ml-frameworks.md) for framework-specific cloud deployment
- Learn about [Monitoring Tools Integration](monitoring-tools.md) for cloud experiment tracking
- Review [CI/CD Integration](ci-cd-pipelines.md) for automated cloud deployment
- Check [Guides](../guides/index) for production cloud deployments
