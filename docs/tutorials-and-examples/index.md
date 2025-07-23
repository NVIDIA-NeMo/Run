---
description: "Comprehensive learning resources for NeMo Run including tutorials, examples, and use cases"
categories: ["tutorials-and-examples"]
tags: ["tutorials", "examples", "use-cases", "learning", "code-samples", "real-world", "step-by-step"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(tutorials-and-examples)=

# NeMo Run Tutorials and Examples

Comprehensive learning resources for NeMo Run, including step-by-step tutorials, complete code examples, and real-world use cases.

## Learning Path

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Tutorials
:link: #tutorials
:link-type: ref
:link-alt: Tutorials Section

Step-by-step learning guides from beginner to advanced
::::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Examples
:link: #examples
:link-type: ref
:link-alt: Examples Section

Complete, runnable code examples for specific scenarios
::::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Use Cases
:link: #use-cases
:link-type: ref
:link-alt: Use Cases Section

Real-world applications and workflows
::::

:::::

## Tutorials {#tutorials}

Learn NeMo Run through hands-on tutorials designed for different skill levels and use cases.

### Beginner Tutorials

Perfect for users new to NeMo Run or ML experiment management.

#### [First Experiment](beginner/first-experiment)
Learn the basics of NeMo Run by creating and running your first ML experiment.

**What you'll learn:**
- Install and configure NeMo Run
- Create your first configuration
- Run a simple training experiment
- Understand basic concepts

**Prerequisites:** Basic Python knowledge, familiarity with ML concepts

#### [Configuration Basics](beginner/configuration-basics)
Master NeMo Run's type-safe configuration system.

**What you'll learn:**
- Use `run.Config` for type-safe configurations
- Create nested configuration structures
- Validate configurations at runtime
- Export configurations to different formats

**Prerequisites:** First Experiment tutorial

#### [Local Execution](beginner/local-execution)
Run experiments on your local machine with proper resource management.

**What you'll learn:**
- Use `run.LocalExecutor` effectively
- Manage GPU resources locally
- Handle environment variables and dependencies
- Debug experiments locally

**Prerequisites:** Configuration Basics tutorial

### Intermediate Tutorials

For users ready to explore distributed computing and advanced features.

#### [Distributed Training](intermediate/distributed-training)
Scale your experiments across multiple nodes and GPUs.

**What you'll learn:**
- Configure multi-node training
- Use Slurm and Kubernetes executors
- Implement distributed data loading
- Monitor distributed experiments

**Prerequisites:** Local Execution tutorial, basic understanding of distributed computing

#### [Hyperparameter Tuning](intermediate/hyperparameter-tuning)
Automate hyperparameter optimization with NeMo Run.

**What you'll learn:**
- Set up hyperparameter sweeps
- Use different search strategies
- Analyze and compare results
- Optimize search efficiency

**Prerequisites:** Distributed Training tutorial

#### [Experiment Management](intermediate/experiment-management)
Master experiment lifecycle management and reproducibility.

**What you'll learn:**
- Organize experiments with metadata
- Track experiment dependencies
- Reproduce past experiments
- Share experiments with team members

**Prerequisites:** Hyperparameter Tuning tutorial

### Advanced Tutorials

For experienced users building production systems.

#### [Production Pipelines](advanced/production-pipelines)
Build robust, scalable ML pipelines for production environments.

**What you'll learn:**
- Design fault-tolerant pipelines
- Implement monitoring and alerting
- Handle data versioning and lineage
- Deploy models with confidence

**Prerequisites:** Experiment Management tutorial

#### [Custom Executors](advanced/custom-executors)
Extend NeMo Run with custom execution environments.

**What you'll learn:**
- Create custom executor classes
- Integrate with new platforms
- Implement custom packaging strategies
- Add platform-specific optimizations

**Prerequisites:** Production Pipelines tutorial

#### [Fault Tolerance](advanced/fault-tolerance)
Build resilient systems that handle failures gracefully.

**What you'll learn:**
- Implement automatic retry logic
- Handle partial failures
- Design checkpoint and recovery systems
- Monitor system health

**Prerequisites:** Custom Executors tutorial

## Examples {#examples}

Complete, runnable examples for common NeMo Run use cases and scenarios.

### ML Framework Examples

Integration examples with popular machine learning frameworks.

#### [PyTorch Training](ml-frameworks/pytorch-training)
Complete PyTorch training pipeline with NeMo Run.

**Features:**
- Distributed training with PyTorch
- Custom model architectures
- Data loading and preprocessing
- Training monitoring and logging

**Use Cases:**
- Computer vision models
- Natural language processing
- Custom neural network architectures

#### [TensorFlow Training](ml-frameworks/tensorflow-training)
TensorFlow integration with NeMo Run for scalable training.

**Features:**
- TensorFlow/Keras model training
- Multi-GPU training strategies
- Custom training loops
- Model checkpointing

**Use Cases:**
- Image classification
- Sequence modeling
- Transfer learning

#### [Hugging Face Transformers](ml-frameworks/huggingface-transformers)
Fine-tune pre-trained models with Hugging Face and NeMo Run.

**Features:**
- Pre-trained model fine-tuning
- Custom datasets
- Training arguments configuration
- Model evaluation

**Use Cases:**
- Text classification
- Named entity recognition
- Question answering
- Text generation

### Cloud Platform Examples

Deploy and run experiments on various cloud platforms.

#### [AWS EC2](cloud-platforms/aws-ec2)
Run NeMo Run experiments on AWS EC2 instances.

**Features:**
- EC2 instance provisioning
- GPU-enabled instances
- Cost optimization
- Security best practices

**Use Cases:**
- Development and testing
- Small to medium-scale training
- Cost-effective experimentation

#### [GCP GKE](cloud-platforms/gcp-gke)
Deploy NeMo Run on Google Kubernetes Engine.

**Features:**
- Kubernetes cluster management
- Auto-scaling capabilities
- GPU node pools
- Persistent storage

**Use Cases:**
- Production ML pipelines
- Large-scale distributed training
- Multi-tenant environments

#### [Azure AKS](cloud-platforms/azure-aks)
Run experiments on Azure Kubernetes Service.

**Features:**
- Azure-specific optimizations
- Managed Kubernetes
- GPU node pools
- Integration with Azure ML

**Use Cases:**
- Enterprise ML workflows
- Hybrid cloud deployments
- Azure ecosystem integration

### Real-World Examples

Production-ready examples for common ML scenarios.

#### [LLM Fine-tuning](real-world/llm-finetuning)
Fine-tune large language models with NeMo Run.

**Features:**
- Parameter-efficient fine-tuning
- LoRA and QLoRA techniques
- Custom datasets
- Evaluation metrics

**Use Cases:**
- Domain adaptation
- Task-specific fine-tuning
- Instruction following

#### [Computer Vision](real-world/computer-vision)
Train computer vision models with NeMo Run.

**Features:**
- Image classification
- Object detection
- Semantic segmentation
- Data augmentation

**Use Cases:**
- Medical imaging
- Autonomous vehicles
- Quality inspection

#### [Recommendation Systems](real-world/recommendation-systems)
Build recommendation systems with NeMo Run.

**Features:**
- Collaborative filtering
- Content-based filtering
- Hybrid approaches
- A/B testing

**Use Cases:**
- E-commerce recommendations
- Content recommendations
- Personalized experiences

## Use Cases {#use-cases}

Real-world applications and workflows demonstrating how NeMo Run solves practical ML challenges.

### Research Use Cases

Academic and industrial research applications of NeMo Run.

#### [Reproducible Research](use-cases/research/reproducible-research)
Ensure complete reproducibility of ML experiments across different environments.

**Key Features:**
- Complete experiment state capture
- Environment-agnostic execution
- Version-controlled configurations
- Automated reproducibility checks

**Applications:**
- Academic research publications
- Industry research collaborations
- Benchmark comparisons
- Method validation

#### [Hyperparameter Optimization](use-cases/research/hyperparameter-optimization)
Automate hyperparameter search with advanced optimization strategies.

**Key Features:**
- Multi-objective optimization
- Bayesian optimization
- Early stopping strategies
- Resource-aware scheduling

**Applications:**
- Model architecture search
- Training hyperparameter tuning
- Automated ML pipelines
- Research methodology validation

#### [Model Comparison](use-cases/research/model-comparison)
Systematically compare different model architectures and approaches.

**Key Features:**
- Standardized evaluation metrics
- Statistical significance testing
- Visualization and reporting
- Automated comparison workflows

**Applications:**
- Architecture research
- Benchmark studies
- Method evaluation
- Publication preparation

### Production Use Cases

Enterprise and production ML system applications.

#### [ML Pipelines](use-cases/production/ml-pipelines)
Build robust, scalable ML pipelines for production environments.

**Key Features:**
- End-to-end pipeline orchestration
- Fault tolerance and recovery
- Monitoring and alerting
- Automated deployment

**Applications:**
- Recommendation systems
- Computer vision applications
- Natural language processing
- Predictive analytics

#### [Model Deployment](use-cases/production/model-deployment)
Deploy ML models with confidence and monitoring.

**Key Features:**
- Model versioning and rollback
- A/B testing capabilities
- Performance monitoring
- Automated scaling

**Applications:**
- Real-time inference
- Batch prediction services
- Model serving platforms
- Edge deployment

#### [Monitoring](use-cases/production/monitoring)
Monitor ML systems and detect issues proactively.

**Key Features:**
- Real-time metrics collection
- Anomaly detection
- Performance tracking
- Automated alerting

**Applications:**
- Model drift detection
- System health monitoring
- Performance optimization
- Incident response

### Collaboration Use Cases

Team collaboration and knowledge sharing workflows.

#### [Team Workflows](use-cases/collaboration/team-workflows)
Enable effective collaboration across ML teams.

**Key Features:**
- Shared experiment repositories
- Knowledge transfer mechanisms
- Standardized workflows
- Team coordination tools

**Applications:**
- Research team collaboration
- Cross-functional ML teams
- Academic-industry partnerships
- Open source contributions

#### [Code Sharing](use-cases/collaboration/code-sharing)
Share and reuse ML code effectively across teams.

**Key Features:**
- Version-controlled configurations
- Reusable components
- Documentation generation
- Code review workflows

**Applications:**
- Internal tool development
- Open source projects
- Research code sharing
- Best practice dissemination

#### [Experiment Tracking](use-cases/collaboration/experiment-tracking)
Track and share experiments across team members.

**Key Features:**
- Centralized experiment management
- Metadata organization
- Result sharing
- Collaboration tools

**Applications:**
- Team experiment coordination
- Knowledge management
- Research documentation
- Training and onboarding

## Getting Started

### Choose Your Path

1. **New to NeMo Run?** Start with [Beginner Tutorials](#tutorials)
2. **Looking for specific examples?** Browse [Examples](#examples) by framework or platform
3. **Building production systems?** Explore [Use Cases](#use-cases) for real-world patterns

### Prerequisites

Before diving in, ensure you have:
- NeMo Run installed (see [Installation Guide](../get-started/install))
- Required dependencies for your chosen examples
- Access to computing resources (local, cloud, or cluster)
- Basic understanding of the target framework or platform

### Learning Progression

**Beginner Path:**
1. [First Experiment](beginner/first-experiment) → [Configuration Basics](beginner/configuration-basics) → [Local Execution](beginner/local-execution)

**Intermediate Path:**
2. [Distributed Training](intermediate/distributed-training) → [Hyperparameter Tuning](intermediate/hyperparameter-tuning) → [Experiment Management](intermediate/experiment-management)

**Advanced Path:**
3. [Production Pipelines](advanced/production-pipelines) → [Custom Executors](advanced/custom-executors) → [Fault Tolerance](advanced/fault-tolerance)

## Features

### Interactive Examples
All tutorials and examples include complete, runnable code that you can execute immediately.

### Progressive Difficulty
Content builds upon each other, ensuring you have the necessary foundation before moving to advanced topics.

### Real-World Scenarios
Examples and use cases are based on actual ML workflows and production use cases.

### Best Practices
Each resource incorporates industry best practices and NeMo Run recommendations.

## Getting Help

- **Stuck on a tutorial?** Check the [Troubleshooting Guide](../reference/troubleshooting)
- **Need more examples?** Explore our [Examples](#examples) section
- **Want to see real use cases?** Visit our [Use Cases](#use-cases) section
- **Have questions?** Review our [FAQs](../reference/faqs)

## Next Steps

After exploring tutorials, examples, and use cases, check out:

- **[Best Practices](../best-practices/index)** - Production-ready patterns and optimizations
- **[Reference Documentation](../reference/index)** - Complete API documentation
- **[Integrations](../integrations/index)** - Platform and tool integrations
- **[Guides](../guides/index)** - Detailed implementation guides
