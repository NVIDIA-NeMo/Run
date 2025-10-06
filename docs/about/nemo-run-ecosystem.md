---
description: "Understanding how NeMo Run integrates with NeMo Framework components and the broader AI development ecosystem"
tags: ["nemo-run", "ecosystem", "nemo-framework", "integration", "architecture", "orchestration"]
categories: ["about"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "text-only"
---

# NeMo Run and the NeMo Ecosystem

The **NeMo ecosystem** is NVIDIA's comprehensive platform for AI model development, spanning from research to production. **NeMo Run** serves as the crucial orchestration layer that bridges model development with scalable execution across diverse computing environments.

## NeMo Run: The Orchestration Framework

**NeMo Run** is a separate orchestration framework that works alongside the NeMo Framework and Libraries. It serves as the crucial bridge between model development and scalable execution, handling all the operational complexities of running AI experiments at scale.

### Core Capabilities

NeMo Run provides four essential orchestration capabilities:

- **Experiment Configuration**: Type-safe, Python-based setup and parameter management that replaces complex YAML files
- **Multi-Environment Execution**: Seamless deployment across local, cluster, and cloud environments without code changes
- **Lifecycle Management**: Comprehensive tracking, logging, and reproducibility features for experiment management
- **Resource Orchestration**: Intelligent packaging, submission, and monitoring of training jobs across diverse infrastructure

### How NeMo Run Integrates with NeMo Components

NeMo Run bridges the gap between model development (NeMo Framework/libraries) and scalable execution across diverse computing environments. **NeMo libraries focus on what to run, while NeMo Run focuses on how to run**—together they provide an end-to-end path from model code to scalable, production-ready execution.

**The Integration Model:**

NeMo Run acts as the "conductor" that takes your model code from NeMo Libraries and makes it runnable at scale. Here's how the integration works:

1. **Model Development**: Use NeMo Libraries (ASR, TTS, LLM, etc.) to define your model architecture and training logic
2. **Configuration**: Use NeMo Run's Python-based configuration system to define experiment parameters and execution requirements
3. **Orchestration**: NeMo Run packages your code, manages dependencies, and handles environment setup
4. **Execution**: Deploy seamlessly across local machines, SLURM clusters, Kubernetes, or cloud platforms
5. **Management**: Track experiments, collect artifacts, and ensure reproducibility across all environments

## NeMo Run's Architecture and Integration

NeMo Run provides the operational bridge from modeling to execution. This section shows how configuration, packaging, and executors turn NeMo library recipes into scalable runs across local machines, clusters, and cloud—while capturing logs, checkpoints, and metadata for reproducibility.

### 1. Modeling Layer (NeMo Framework & Libraries)

- **NeMo Libraries**: Specialized components for speech, language, and multimodal AI
- **Model Implementations**: Pre-built architectures for large language models, ASR, TTS, and multimodal models
- **Training Recipes**: Complete configurations defining model architectures, hyperparameters, and training procedures
- **Data Processing**: Utilities for data curation, tokenization, and preprocessing
- **Output**: Python functions and configurations that define "what to train"

### 2. Orchestration Layer (NeMo Run)

- **Configuration**: `run.Config` / `run.Partial` for type-safe, Python-based experiment setup
- **Packaging**: Intelligent code packaging strategies (git/pattern/hybrid) for reproducible deployments
- **Executors**: Backend-agnostic execution support (Local, Docker, SLURM, Ray, Kubernetes, cloud platforms)
- **Management**: Automated capture of logs, metadata, checkpoints, and artifacts for reproducibility
- **Purpose**: Defines "how to run" experiments across diverse computing environments

### 3. Runtime/Infrastructure Layer

- **Compute Resources**: Local workstations, GPU clusters, HPC systems (SLURM), cloud platforms (AWS, GCP, Azure)
- **Container Platforms**: Docker environments, Kubernetes clusters, managed cloud services
- **Storage Systems**: Distributed file systems, object storage, checkpoint repositories
- **Monitoring & Observability**: Metrics collection, log aggregation, experiment tracking dashboards
- **Resource Management**: Auto scaling, quota management, environment isolation, and cost optimization

<style>
.clickable-diagram {
    cursor: pointer;
    transition: transform 0.2s ease-in-out;
    border: 2px solid #4A90E2;
    border-radius: 8px;
    padding: 10px;
    background: transparent;
}

.clickable-diagram:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
}

.clickable-diagram:active {
    transform: scale(0.98);
}

/* Modal styles for expanded view */
.diagram-modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.8);
    backdrop-filter: blur(5px);
}

.diagram-modal-content {
    position: relative;
    margin: 5% auto;
    padding: 20px;
    width: 90%;
    max-width: 1200px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.diagram-modal-close {
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 28px;
    font-weight: bold;
    color: #666;
    cursor: pointer;
    transition: color 0.2s;
}

.diagram-modal-close:hover {
    color: #333;
}

.diagram-modal img {
    width: 100%;
    height: auto;
    border-radius: 8px;
}

.diagram-modal-description {
    margin-top: 12px;
    color: #444;
    line-height: 1.5;
}

/* Hide auxiliary buttons (Copy / Explain / Ask AI) inside the diagram area and modal */
#ecosystem-diagram .copybtn,
.diagram-modal .copybtn,
#ecosystem-diagram .ai-assistant-button,
.diagram-modal .ai-assistant-button,
#ecosystem-diagram .ai-explain-button,
.diagram-modal .ai-explain-button,
#ecosystem-diagram .ai-toolbar,
.diagram-modal .ai-toolbar,
#ecosystem-diagram .ai-btn,
.diagram-modal .ai-btn {
    display: none !important;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const diagram = document.getElementById('ecosystem-diagram');

    if (diagram) {
        diagram.addEventListener('click', function() {
            // Create modal
            const modal = document.createElement('div');
            modal.className = 'diagram-modal';
            modal.innerHTML = `
                <div class="diagram-modal-content">
                    <span class="diagram-modal-close" aria-label="Close dialog">&times;</span>
                    <h3>NeMo Run Ecosystem Architecture</h3>
                    <div style="max-height: 80vh; overflow-y: auto;">${this.innerHTML}</div>
                    <div class="diagram-modal-description">
                        This diagram illustrates the complete NeMo Run ecosystem architecture. The flow starts with the <strong>Modeling layer</strong> (NeMo Libraries for ASR, TTS, LLM, and multimodal models, along with training recipes and data processing). These components feed into the <strong>NeMo Run Orchestration layer</strong>, which provides type-safe configuration with run.Config/run.Partial, intelligent code packaging, and multiple execution backends. The orchestration layer then deploys to various <strong>Runtime environments</strong> including local workstations, GPU clusters, container platforms, and cloud services. Throughout this flow, the experiment management system captures logs, metadata, and artifacts, ensuring full reproducibility and traceability of ML experiments.
                    </div>
                </div>
            `;

            document.body.appendChild(modal);
            modal.style.display = 'block';

            // Remove unwanted buttons from within the modal (Copy / Explain / Ask AI)
            (function removeUnwantedButtons(root) {
                // Remove entire AI toolbar if present
                root.querySelectorAll('.ai-toolbar').forEach(el => el.remove());
                root.querySelectorAll('.copybtn').forEach(el => el.remove());

                const elements = root.querySelectorAll('button, a');
                elements.forEach((el) => {
                    const label = `${el.getAttribute('aria-label') || ''} ${el.getAttribute('title') || ''} ${el.textContent || ''}`.trim();
                    if (/\b(copy|explain|ask ai)\b/i.test(label) || el.classList.contains('copybtn')) {
                        el.style.display = 'none';
                    }
                });
            })(modal);

            // Also hide/remove toolbars in the inline diagram area
            (function removeFromInlineDiagram() {
                const container = document.getElementById('ecosystem-diagram');
                if (!container) return;
                container.querySelectorAll('.ai-toolbar, .copybtn').forEach(el => el.remove());
                container.querySelectorAll('.ai-btn').forEach(el => el.style.display = 'none');
            })();

            // Close modal functionality
            const closeBtn = modal.querySelector('.diagram-modal-close');
            closeBtn.addEventListener('click', function() {
                modal.remove();
            });

            // Close on outside click
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.remove();
                }
            });

            // Close on Escape key
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    modal.remove();
                }
            });
        });
    }
});
</script>

<div class="clickable-diagram" id="ecosystem-diagram">

![NeMo Run Ecosystem Diagram](../assets/nemo-run-ecosystem.png)

*Click the diagram to view it in full size*

</div>
