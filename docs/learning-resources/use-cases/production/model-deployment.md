---
description: "Deploy ML models with confidence and monitoring using NeMo Run"
categories: ["use-cases", "production"]
tags: ["model-deployment", "production", "monitoring", "a-b-testing", "model-serving"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "use-case"
modality: "text-only"
---

# Model Deployment

Deploy ML models with confidence and monitoring using NeMo Run.

## Overview

Model deployment is a critical phase in the ML lifecycle that requires careful planning, robust infrastructure, and comprehensive monitoring. NeMo Run provides powerful tools for deploying ML models with confidence, including model versioning, A/B testing capabilities, performance monitoring, and automated scaling.

## Key Features

### Model Versioning and Rollback

- Version-controlled model deployments
- Automated rollback mechanisms
- Model registry integration
- Deployment history tracking

### A/B Testing Capabilities

- Traffic splitting between model versions
- Statistical significance testing
- Performance comparison
- Automated winner selection

### Monitor Performance

- Real-time performance metrics
- Model drift detection
- Resource utilization monitoring
- Automated alerting

### Automated Scaling

- Load-based auto-scaling
- Resource optimization
- Cost management
- Performance tuning

## Use Case Scenarios

Examples that illustrate common deployment patterns.

### Scenario 1: Production Model Deployment

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import mlflow
import docker

@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_path: str
    model_version: str
    deployment_name: str
    environment: str
    resources: Dict[str, Any]
    scaling_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]

@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    model_a: DeploymentConfig
    model_b: DeploymentConfig
    traffic_split: Dict[str, float]
    evaluation_metrics: List[str]
    test_duration: int  # days

def deploy_model_to_production():
    """Deploy a trained model to production environment."""

    # Load the trained model
    model_path = "models/best_model_v1.0"

    # Define deployment configuration
    deployment_config = DeploymentConfig(
        model_path=model_path,
        model_version="v1.0",
        deployment_name="image-classifier-prod",
        environment="production",
        resources={
            "cpu": "2",
            "memory": "4Gi",
            "gpu": "1"
        },
        scaling_config={
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70
        },
        monitoring_config={
            "metrics": ["accuracy", "latency", "throughput"],
            "alerts": ["high_latency", "low_accuracy", "high_error_rate"],
            "dashboards": ["performance", "errors", "traffic"]
        }
    )

    # Deploy model
    with run.Experiment("model_deployment") as exp:
        exp.add(deployment_config, name="production_deployment")
        deployment_results = exp.run()

    return deployment_results

def deploy_with_rollback_support():
    """Deploy model with automatic rollback capabilities."""

    # Define deployment with rollback
    deployment_config = DeploymentConfig(
        model_path="models/model_v2.0",
        model_version="v2.0",
        deployment_name="image-classifier-v2",
        environment="production",
        resources={"cpu": "2", "memory": "4Gi"},
        scaling_config={"min_replicas": 2, "max_replicas": 10},
        monitoring_config={
            "rollback_triggers": ["accuracy_drop", "latency_increase", "error_rate_spike"],
            "health_checks": ["model_loading", "prediction_quality", "resource_usage"]
        }
    )

    # Deploy with rollback monitoring
    with run.Experiment("deployment_with_rollback") as exp:
        exp.add(deployment_config, name="v2_deployment")

        # Add rollback configuration
        exp.add({
            "rollback_model": "models/model_v1.0",
            "rollback_conditions": {
                "accuracy_threshold": 0.95,
                "latency_threshold": 100,  # ms
                "error_rate_threshold": 0.01
            },
            "rollback_automation": True
        }, name="rollback_config")

        deployment_results = exp.run()

    return deployment_results
```

### Scenario 2: A/B Testing Deployment

```python
def deploy_ab_test():
    """Deploy A/B test between two model versions."""

    # Define model A (current production)
    model_a_config = DeploymentConfig(
        model_path="models/current_production",
        model_version="v1.0",
        deployment_name="model-a",
        environment="production",
        resources={"cpu": "2", "memory": "4Gi"},
        scaling_config={"min_replicas": 1, "max_replicas": 5},
        monitoring_config={"metrics": ["accuracy", "latency"]}
    )

    # Define model B (new candidate)
    model_b_config = DeploymentConfig(
        model_path="models/new_candidate",
        model_version="v2.0",
        deployment_name="model-b",
        environment="production",
        resources={"cpu": "2", "memory": "4Gi"},
        scaling_config={"min_replicas": 1, "max_replicas": 5},
        monitoring_config={"metrics": ["accuracy", "latency"]}
    )

    # Define A/B test configuration
    ab_test_config = ABTestConfig(
        model_a=model_a_config,
        model_b=model_b_config,
        traffic_split={"model_a": 0.5, "model_b": 0.5},
        evaluation_metrics=["accuracy", "latency", "user_satisfaction"],
        test_duration=14  # 14 days
    )

    # Deploy A/B test
    with run.Experiment("ab_test_deployment") as exp:
        exp.add(ab_test_config, name="ab_test_config")

        # Add traffic routing
        exp.add({
            "routing_strategy": "weighted_random",
            "traffic_split": ab_test_config.traffic_split,
            "evaluation_criteria": ab_test_config.evaluation_metrics
        }, name="traffic_routing")

        # Add monitoring and evaluation
        exp.add({
            "monitoring_interval": "1h",
            "statistical_tests": ["t_test", "mann_whitney"],
            "winner_selection_criteria": {
                "accuracy_improvement": 0.02,
                "latency_improvement": 0.1,
                "statistical_significance": 0.05
            }
        }, name="evaluation_config")

        ab_test_results = exp.run()

    return ab_test_results

def canary_deployment():
    """Deploy new model using canary deployment strategy."""

    # Define canary deployment stages
    canary_stages = [
        {"traffic_percentage": 0.01, "duration": "1h"},   # 1% for 1 hour
        {"traffic_percentage": 0.05, "duration": "6h"},   # 5% for 6 hours
        {"traffic_percentage": 0.10, "duration": "12h"},  # 10% for 12 hours
        {"traffic_percentage": 0.25, "duration": "24h"},  # 25% for 24 hours
        {"traffic_percentage": 0.50, "duration": "48h"},  # 50% for 48 hours
        {"traffic_percentage": 1.00, "duration": "168h"}  # 100% for 1 week
    ]

    # Define new model configuration
    new_model_config = DeploymentConfig(
        model_path="models/new_model",
        model_version="v2.0",
        deployment_name="canary-deployment",
        environment="production",
        resources={"cpu": "2", "memory": "4Gi"},
        scaling_config={"min_replicas": 1, "max_replicas": 10},
        monitoring_config={
            "health_checks": ["response_time", "error_rate", "accuracy"],
            "rollback_triggers": ["high_error_rate", "high_latency", "low_accuracy"]
        }
    )

    # Deploy with canary strategy
    with run.Experiment("canary_deployment") as exp:
        exp.add(new_model_config, name="new_model")

        # Add canary configuration
        exp.add({
            "stages": canary_stages,
            "promotion_criteria": {
                "max_error_rate": 0.01,
                "max_latency": 100,
                "min_accuracy": 0.95
            },
            "rollback_criteria": {
                "error_rate_threshold": 0.05,
                "latency_threshold": 200,
                "accuracy_threshold": 0.90
            }
        }, name="canary_config")

        canary_results = exp.run()

    return canary_results
```

### Scenario 3: Multi-Model Deployment

```python
def deploy_ensemble_model():
    """Deploy ensemble of multiple models."""

    # Define ensemble configuration
    ensemble_config = {
        "models": [
            {
                "name": "model_1",
                "path": "models/random_forest",
                "weight": 0.4,
                "type": "classification"
            },
            {
                "name": "model_2",
                "path": "models/neural_network",
                "weight": 0.4,
                "type": "classification"
            },
            {
                "name": "model_3",
                "path": "models/gradient_boosting",
                "weight": 0.2,
                "type": "classification"
            }
        ],
        "ensemble_strategy": "weighted_voting",
        "deployment_config": {
            "name": "ensemble-classifier",
            "environment": "production",
            "resources": {"cpu": "4", "memory": "8Gi"},
            "scaling": {"min_replicas": 2, "max_replicas": 10}
        }
    }

    # Deploy ensemble
    with run.Experiment("ensemble_deployment") as exp:
        exp.add(ensemble_config, name="ensemble_config")

        # Add ensemble routing logic
        exp.add({
            "routing_logic": "weighted_ensemble",
            "fallback_strategy": "majority_voting",
            "performance_monitoring": {
                "individual_model_metrics": True,
                "ensemble_metrics": True,
                "model_contribution_tracking": True
            }
        }, name="ensemble_routing")

        ensemble_results = exp.run()

    return ensemble_results

def deploy_model_pipeline():
    """Deploy a complete ML pipeline with multiple stages."""

    # Define pipeline stages
    pipeline_config = {
        "stages": [
            {
                "name": "data_preprocessing",
                "type": "data_processing",
                "config": {
                    "input_format": "json",
                    "output_format": "tensor",
                    "preprocessing_steps": ["normalization", "encoding"]
                }
            },
            {
                "name": "feature_extraction",
                "type": "feature_engineering",
                "config": {
                    "feature_extractors": ["cnn", "handcrafted"],
                    "output_dimensions": 512
                }
            },
            {
                "name": "classification",
                "type": "model_inference",
                "config": {
                    "model_path": "models/classifier",
                    "output_format": "probabilities"
                }
            },
            {
                "name": "post_processing",
                "type": "result_processing",
                "config": {
                    "threshold": 0.5,
                    "output_format": "json"
                }
            }
        ],
        "deployment_config": {
            "name": "ml-pipeline",
            "environment": "production",
            "resources": {"cpu": "8", "memory": "16Gi"},
            "scaling": {"min_replicas": 3, "max_replicas": 20}
        }
    }

    # Deploy pipeline
    with run.Experiment("pipeline_deployment") as exp:
        exp.add(pipeline_config, name="pipeline_config")

        # Add pipeline monitoring
        exp.add({
            "stage_monitoring": True,
            "end_to_end_monitoring": True,
            "performance_tracking": {
                "latency_per_stage": True,
                "throughput_per_stage": True,
                "error_tracking": True
            }
        }, name="pipeline_monitoring")

        pipeline_results = exp.run()

    return pipeline_results
```

## Advanced Deployment Strategies

Progressively safer rollout approaches for real systems.

### 1. Blue-Green Deployment

```python
def blue_green_deployment():
    """Implement blue-green deployment strategy."""

    # Define blue (current) and green (new) environments
    blue_config = DeploymentConfig(
        model_path="models/current_production",
        model_version="v1.0",
        deployment_name="blue-environment",
        environment="production",
        resources={"cpu": "2", "memory": "4Gi"},
        scaling_config={"min_replicas": 2, "max_replicas": 10}
    )

    green_config = DeploymentConfig(
        model_path="models/new_version",
        model_version="v2.0",
        deployment_name="green-environment",
        environment="staging",
        resources={"cpu": "2", "memory": "4Gi"},
        scaling_config={"min_replicas": 2, "max_replicas": 10}
    )

    # Deploy blue-green strategy
    with run.Experiment("blue_green_deployment") as exp:
        exp.add(blue_config, name="blue_environment")
        exp.add(green_config, name="green_environment")

        # Add traffic switching logic
        exp.add({
            "switch_strategy": "gradual",
            "switch_stages": [
                {"green_traffic": 0.1, "duration": "1h"},
                {"green_traffic": 0.5, "duration": "6h"},
                {"green_traffic": 1.0, "duration": "24h"}
            ],
            "rollback_conditions": {
                "error_rate": 0.05,
                "latency_increase": 0.2
            }
        }, name="traffic_switching")

        deployment_results = exp.run()

    return deployment_results
```

### 2. Multi-Region Deployment

```python
def multi_region_deployment():
    """Deploy model across multiple regions for global availability."""

    regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]

    deployment_configs = {}

    for region in regions:
        deployment_configs[region] = DeploymentConfig(
            model_path="models/global_model",
            model_version="v1.0",
            deployment_name=f"model-{region}",
            environment="production",
            resources={"cpu": "2", "memory": "4Gi"},
            scaling_config={"min_replicas": 2, "max_replicas": 10}
        )

    # Deploy to multiple regions
    with run.Experiment("multi_region_deployment") as exp:
        for region, config in deployment_configs.items():
            exp.add(config, name=f"deployment_{region}")

        # Add global load balancing
        exp.add({
            "load_balancer": "global",
            "routing_strategy": "geographic",
            "health_checks": "cross_region",
            "failover_strategy": "automatic"
        }, name="global_routing")

        multi_region_results = exp.run()

    return multi_region_results
```

### 3. Edge Deployment

```python
def edge_deployment():
    """Deploy model to edge devices for low-latency inference."""

    # Define edge deployment configuration
    edge_config = {
        "edge_devices": [
            {
                "device_type": "iot_gateway",
                "location": "factory_floor_1",
                "resources": {"cpu": "1", "memory": "2Gi"},
                "model_config": {
                    "model_path": "models/edge_optimized",
                    "quantization": True,
                    "pruning": True
                }
            },
            {
                "device_type": "edge_server",
                "location": "data_center_edge",
                "resources": {"cpu": "4", "memory": "8Gi"},
                "model_config": {
                    "model_path": "models/edge_server_model",
                    "quantization": False,
                    "pruning": False
                }
            }
        ],
        "deployment_strategy": "distributed",
        "sync_strategy": "periodic",
        "offline_capability": True
    }

    # Deploy to edge devices
    with run.Experiment("edge_deployment") as exp:
        exp.add(edge_config, name="edge_config")

        # Add edge-specific monitoring
        exp.add({
            "edge_monitoring": {
                "device_health": True,
                "model_performance": True,
                "network_connectivity": True,
                "battery_usage": True
            },
            "sync_config": {
                "sync_interval": "1h",
                "model_updates": "incremental",
                "data_collection": "periodic"
            }
        }, name="edge_monitoring")

        edge_results = exp.run()

    return edge_results
```

## Monitoring and Observability

Keep deployments healthy with proactive monitoring.

### 1. Real-Time Performance Monitoring

```python
def setup_performance_monitoring():
    """Set up comprehensive performance monitoring for deployed models."""

    monitoring_config = {
        "metrics_collection": {
            "model_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "system_metrics": ["cpu_usage", "memory_usage", "gpu_usage"],
            "business_metrics": ["user_satisfaction", "conversion_rate"],
            "custom_metrics": ["prediction_confidence", "feature_importance"]
        },
        "alerting": {
            "accuracy_drop": {"threshold": 0.02, "window": "1h"},
            "latency_increase": {"threshold": 0.5, "window": "5m"},
            "error_rate_spike": {"threshold": 0.05, "window": "10m"},
            "resource_exhaustion": {"threshold": 0.9, "window": "5m"}
        },
        "dashboards": {
            "real_time": ["performance", "errors", "traffic"],
            "historical": ["trends", "comparisons", "anomalies"],
            "business": ["impact", "roi", "user_behavior"]
        }
    }

    # Set up monitoring
    with run.Experiment("performance_monitoring") as exp:
        exp.add(monitoring_config, name="monitoring_config")

        # Add monitoring agents
        exp.add({
            "monitoring_agents": {
                "model_monitor": True,
                "system_monitor": True,
                "business_monitor": True
            },
            "data_collection": {
                "sampling_rate": 0.1,
                "storage": "time_series_db",
                "retention": "30d"
            }
        }, name="monitoring_agents")

        monitoring_results = exp.run()

    return monitoring_results
```

### 2. Model Drift Detection

```python
def setup_drift_detection():
    """Set up model drift detection and alerting."""

    drift_config = {
        "drift_detection": {
            "data_drift": {
                "methods": ["ks_test", "chi_square", "wasserstein"],
                "features": ["all"],
                "frequency": "daily",
                "threshold": 0.05
            },
            "concept_drift": {
                "methods": ["performance_monitoring", "distribution_shift"],
                "window_size": "7d",
                "threshold": 0.1
            },
            "label_drift": {
                "methods": ["distribution_comparison"],
                "frequency": "weekly",
                "threshold": 0.05
            }
        },
        "mitigation_strategies": {
            "retraining": {
                "trigger": "drift_detected",
                "strategy": "incremental",
                "data_window": "30d"
            },
            "fallback": {
                "model": "previous_version",
                "conditions": ["high_drift", "performance_degradation"]
            }
        }
    }

    # Set up drift detection
    with run.Experiment("drift_detection") as exp:
        exp.add(drift_config, name="drift_config")

        # Add drift monitoring
        exp.add({
            "monitoring_pipeline": {
                "data_collection": "real_time",
                "drift_analysis": "automated",
                "alerting": "immediate",
                "reporting": "daily"
            }
        }, name="drift_monitoring")

        drift_results = exp.run()

    return drift_results
```

### 3. Automated Scaling

```python
def setup_auto_scaling():
    """Set up automated scaling based on load and performance."""

    scaling_config = {
        "scaling_policies": {
            "cpu_based": {
                "metric": "cpu_utilization",
                "threshold": 70,
                "scale_up": {"increment": 1, "cooldown": "5m"},
                "scale_down": {"increment": 1, "cooldown": "10m"}
            },
            "memory_based": {
                "metric": "memory_utilization",
                "threshold": 80,
                "scale_up": {"increment": 1, "cooldown": "5m"},
                "scale_down": {"increment": 1, "cooldown": "10m"}
            },
            "latency_based": {
                "metric": "response_time",
                "threshold": 100,  # ms
                "scale_up": {"increment": 2, "cooldown": "2m"},
                "scale_down": {"increment": 1, "cooldown": "5m"}
            },
            "throughput_based": {
                "metric": "requests_per_second",
                "threshold": 1000,
                "scale_up": {"increment": 1, "cooldown": "3m"},
                "scale_down": {"increment": 1, "cooldown": "10m"}
            }
        },
        "resource_limits": {
            "min_replicas": 2,
            "max_replicas": 20,
            "max_cpu": "8",
            "max_memory": "16Gi"
        }
    }

    # Set up auto scaling
    with run.Experiment("auto_scaling") as exp:
        exp.add(scaling_config, name="scaling_config")

        # Add scaling monitoring
        exp.add({
            "scaling_monitoring": {
                "scale_events": True,
                "performance_impact": True,
                "cost_tracking": True
            }
        }, name="scaling_monitoring")

        scaling_results = exp.run()

    return scaling_results
```

## Security and Compliance

Harden deployments and meet regulatory needs.

### 1. Model Security

```python
def setup_model_security():
    """Set up security measures for deployed models."""

    security_config = {
        "authentication": {
            "api_keys": True,
            "oauth2": True,
            "rate_limiting": {"requests_per_minute": 1000}
        },
        "authorization": {
            "role_based_access": True,
            "model_access_control": True,
            "data_access_control": True
        },
        "encryption": {
            "data_in_transit": "tls_1_3",
            "data_at_rest": "aes_256",
            "model_encryption": True
        },
        "auditing": {
            "access_logs": True,
            "prediction_logs": True,
            "model_changes": True
        }
    }

    # Set up security
    with run.Experiment("model_security") as exp:
        exp.add(security_config, name="security_config")

        # Add security monitoring
        exp.add({
            "security_monitoring": {
                "threat_detection": True,
                "anomaly_detection": True,
                "compliance_reporting": True
            }
        }, name="security_monitoring")

        security_results = exp.run()

    return security_results
```

### 2. Compliance and Governance

```python
def setup_compliance_monitoring():
    """Set up compliance monitoring for regulatory requirements."""

    compliance_config = {
        "data_privacy": {
            "gdpr_compliance": True,
            "data_retention": "90d",
            "data_anonymization": True,
            "consent_management": True
        },
        "model_governance": {
            "model_lineage": True,
            "version_control": True,
            "approval_workflow": True,
            "audit_trail": True
        },
        "performance_monitoring": {
            "bias_detection": True,
            "fairness_metrics": True,
            "explainability": True,
            "transparency": True
        }
    }

    # Set up compliance monitoring
    with run.Experiment("compliance_monitoring") as exp:
        exp.add(compliance_config, name="compliance_config")

        # Add compliance reporting
        exp.add({
            "compliance_reporting": {
                "automated_reports": True,
                "regulatory_submissions": True,
                "audit_support": True
            }
        }, name="compliance_reporting")

        compliance_results = exp.run()

    return compliance_results
```

## Best Practices

Checklists and tips to improve deployment quality.

### 1. Deployment Checklist

```python
def deployment_checklist():
    """Comprehensive deployment checklist."""

    checklist = {
        "pre_deployment": [
            "Model validation completed",
            "Performance benchmarks established",
            "Security review completed",
            "Compliance requirements verified",
            "Rollback plan prepared",
            "Monitoring configured",
            "Team notified"
        ],
        "deployment": [
            "Health checks passing",
            "Traffic routing configured",
            "Monitoring active",
            "Alerts configured",
            "Documentation updated"
        ],
        "post_deployment": [
            "Performance monitoring",
            "Error rate monitoring",
            "User feedback collection",
            "Cost monitoring",
            "Regular health checks"
        ]
    }

    return checklist
```

### 2. Performance Optimization

```python
def optimize_deployment_performance():
    """Optimize deployment for performance and cost."""

    optimization_config = {
        "model_optimization": {
            "quantization": True,
            "pruning": True,
            "compression": True,
            "batch_processing": True
        },
        "infrastructure_optimization": {
            "resource_rightsizing": True,
            "auto_scaling": True,
            "load_balancing": True,
            "caching": True
        },
        "cost_optimization": {
            "spot_instances": True,
            "reserved_instances": True,
            "cost_monitoring": True,
            "budget_alerts": True
        }
    }

    return optimization_config
```

## Summary

NeMo Run provides comprehensive tools for model deployment that enable teams to:

- **Deploy with Confidence**: Version control, rollback mechanisms, and health checks
- **Test Safely**: A/B testing, canary deployments, and gradual rollouts
- **Monitor Effectively**: Real-time monitoring, drift detection, and automated alerting
- **Scale Automatically**: Load-based scaling, resource optimization, and cost management
- **Ensure Security**: Authentication, authorization, encryption, and compliance

By leveraging these capabilities, organizations can deploy ML models safely and efficiently while maintaining high performance and reliability standards.
