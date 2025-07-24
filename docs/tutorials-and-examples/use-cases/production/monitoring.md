---
description: "Monitor ML systems and detect issues proactively using NeMo Run"
categories: ["use-cases", "production"]
tags: ["monitoring", "production", "alerting", "anomaly-detection", "performance-tracking"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "use-case"
modality: "text-only"
---

# Monitoring

Monitor ML systems and detect issues proactively using NeMo Run.

## Overview

Effective monitoring is crucial for maintaining reliable ML systems in production. NeMo Run provides comprehensive monitoring capabilities including real-time metrics collection, anomaly detection, performance tracking, and automated alerting to ensure ML systems operate optimally and issues are detected and resolved quickly.

## Key Features

### Real-Time Metrics Collection
- Model performance metrics
- System resource utilization
- Business impact metrics
- Custom application metrics

### Anomaly Detection
- Statistical anomaly detection
- Machine learning-based detection
- Threshold-based alerting
- Trend analysis

### Performance Tracking
- Response time monitoring
- Throughput tracking
- Resource utilization
- Cost monitoring

### Automated Alerting
- Multi-channel notifications
- Escalation procedures
- Incident management
- On-call integration

## Use Case Scenarios

### Scenario 1: Model Performance Monitoring

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass
class MonitoringConfig:
    """Configuration for ML system monitoring."""
    model_name: str
    metrics: List[str]
    thresholds: Dict[str, float]
    alert_channels: List[str]
    monitoring_interval: str

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    detection_methods: List[str]
    sensitivity: float
    window_size: str
    training_period: str

def setup_model_performance_monitoring():
    """Set up comprehensive model performance monitoring."""

    # Define monitoring configuration
    monitoring_config = MonitoringConfig(
        model_name="image-classifier",
        metrics=[
            "accuracy", "precision", "recall", "f1_score",
            "latency", "throughput", "error_rate"
        ],
        thresholds={
            "accuracy": 0.95,
            "latency": 100,  # ms
            "error_rate": 0.01,
            "throughput": 1000  # requests per second
        },
        alert_channels=["email", "slack", "pagerduty"],
        monitoring_interval="1m"
    )

    # Set up monitoring
    with run.Experiment("model_performance_monitoring") as exp:
        exp.add(monitoring_config, name="monitoring_config")

        # Add metric collection
        exp.add({
            "data_collection": {
                "sampling_rate": 1.0,
                "storage": "time_series_db",
                "retention": "90d"
            },
            "real_time_processing": True
        }, name="data_collection")

        # Add alerting configuration
        exp.add({
            "alerting": {
                "threshold_alerts": True,
                "trend_alerts": True,
                "anomaly_alerts": True
            }
        }, name="alerting_config")

        monitoring_results = exp.run()

    return monitoring_results

def monitor_model_drift():
    """Monitor for model drift and performance degradation."""

    drift_config = {
        "drift_detection": {
            "data_drift": {
                "methods": ["ks_test", "chi_square", "wasserstein"],
                "features": ["all"],
                "frequency": "daily",
                "threshold": 0.05
            },
            "performance_drift": {
                "metrics": ["accuracy", "precision", "recall"],
                "window_size": "7d",
                "detection_method": "statistical_test",
                "threshold": 0.02
            },
            "concept_drift": {
                "detection_method": "performance_monitoring",
                "window_size": "14d",
                "threshold": 0.1
            }
        },
        "mitigation": {
            "automatic_retraining": True,
            "fallback_model": "previous_version",
            "alert_on_drift": True
        }
    }

    # Set up drift monitoring
    with run.Experiment("drift_monitoring") as exp:
        exp.add(drift_config, name="drift_config")

        # Add drift analysis
        exp.add({
            "analysis_pipeline": {
                "data_collection": "continuous",
                "drift_analysis": "automated",
                "reporting": "daily"
            }
        }, name="drift_analysis")

        drift_results = exp.run()

    return drift_results
```

### Scenario 2: System Health Monitoring

```python
def monitor_system_health():
    """Monitor system health and resource utilization."""

    system_monitoring_config = {
        "infrastructure_metrics": {
            "cpu_usage": {"threshold": 80, "alert": True},
            "memory_usage": {"threshold": 85, "alert": True},
            "gpu_usage": {"threshold": 90, "alert": True},
            "disk_usage": {"threshold": 85, "alert": True},
            "network_io": {"threshold": 1000, "alert": True}  # MB/s
        },
        "application_metrics": {
            "response_time": {"threshold": 100, "alert": True},  # ms
            "error_rate": {"threshold": 0.01, "alert": True},
            "throughput": {"threshold": 1000, "alert": False},  # req/s
            "queue_length": {"threshold": 100, "alert": True}
        },
        "business_metrics": {
            "user_satisfaction": {"threshold": 0.8, "alert": True},
            "conversion_rate": {"threshold": 0.05, "alert": True},
            "revenue_impact": {"threshold": -0.1, "alert": True}
        }
    }

    # Set up system monitoring
    with run.Experiment("system_health_monitoring") as exp:
        exp.add(system_monitoring_config, name="system_monitoring")

        # Add health checks
        exp.add({
            "health_checks": {
                "model_loading": {"frequency": "5m", "timeout": "30s"},
                "prediction_quality": {"frequency": "1m", "sample_size": 100},
                "resource_availability": {"frequency": "1m"},
                "network_connectivity": {"frequency": "30s"}
            }
        }, name="health_checks")

        # Add alerting
        exp.add({
            "alerting": {
                "channels": ["email", "slack", "pagerduty"],
                "escalation": {
                    "level_1": {"timeout": "5m", "notify": ["oncall"]},
                    "level_2": {"timeout": "15m", "notify": ["manager"]},
                    "level_3": {"timeout": "1h", "notify": ["emergency"]}
                }
            }
        }, name="alerting")

        system_results = exp.run()

    return system_results

def monitor_cost_and_efficiency():
    """Monitor cost and resource efficiency."""

    cost_monitoring_config = {
        "cost_metrics": {
            "compute_cost": {"budget": 1000, "alert": True},  # $/day
            "storage_cost": {"budget": 100, "alert": True},   # $/day
            "network_cost": {"budget": 50, "alert": True},    # $/day
            "total_cost": {"budget": 1200, "alert": True}     # $/day
        },
        "efficiency_metrics": {
            "requests_per_dollar": {"threshold": 1000, "alert": False},
            "latency_per_dollar": {"threshold": 0.1, "alert": True},  # ms/$
            "accuracy_per_dollar": {"threshold": 0.001, "alert": False}
        },
        "optimization": {
            "auto_scaling": True,
            "resource_rightsizing": True,
            "cost_alerts": True
        }
    }

    # Set up cost monitoring
    with run.Experiment("cost_monitoring") as exp:
        exp.add(cost_monitoring_config, name="cost_monitoring")

        # Add cost optimization
        exp.add({
            "optimization": {
                "spot_instances": True,
                "reserved_instances": True,
                "cost_tracking": True
            }
        }, name="cost_optimization")

        cost_results = exp.run()

    return cost_results
```

### Scenario 3: Anomaly Detection

```python
def setup_anomaly_detection():
    """Set up comprehensive anomaly detection for ML systems."""

    anomaly_config = AnomalyDetectionConfig(
        detection_methods=["statistical", "ml_based", "threshold"],
        sensitivity=0.8,
        window_size="1h",
        training_period="30d"
    )

    # Define anomaly detection strategies
    anomaly_strategies = {
        "statistical_detection": {
            "methods": ["z_score", "iqr", "isolation_forest"],
            "window_size": "1h",
            "sensitivity": 0.8
        },
        "ml_based_detection": {
            "models": ["autoencoder", "lstm", "random_forest"],
            "training_data": "30d",
            "update_frequency": "daily"
        },
        "threshold_detection": {
            "static_thresholds": True,
            "dynamic_thresholds": True,
            "adaptive_thresholds": True
        }
    }

    # Set up anomaly detection
    with run.Experiment("anomaly_detection") as exp:
        exp.add(anomaly_config, name="anomaly_config")
        exp.add(anomaly_strategies, name="detection_strategies")

        # Add anomaly response
        exp.add({
            "response_actions": {
                "immediate": ["alert", "log"],
                "short_term": ["investigate", "mitigate"],
                "long_term": ["analyze", "improve"]
            }
        }, name="response_actions")

        anomaly_results = exp.run()

    return anomaly_results

def detect_performance_anomalies():
    """Detect performance anomalies in real-time."""

    performance_anomaly_config = {
        "metrics_to_monitor": [
            "response_time", "throughput", "error_rate",
            "accuracy", "latency", "cpu_usage"
        ],
        "detection_methods": {
            "response_time": {
                "method": "statistical",
                "baseline": "7d_rolling_mean",
                "threshold": 2.0  # standard deviations
            },
            "error_rate": {
                "method": "threshold",
                "baseline": 0.01,
                "threshold": 0.05
            },
            "accuracy": {
                "method": "trend",
                "baseline": "7d_rolling_mean",
                "threshold": -0.02
            }
        },
        "alerting": {
            "immediate": ["response_time_spike", "error_rate_spike"],
            "urgent": ["accuracy_drop", "throughput_drop"],
            "warning": ["trend_changes", "pattern_shifts"]
        }
    }

    # Set up performance anomaly detection
    with run.Experiment("performance_anomaly_detection") as exp:
        exp.add(performance_anomaly_config, name="performance_anomaly_config")

        # Add real-time processing
        exp.add({
            "real_time_processing": {
                "stream_processing": True,
                "window_size": "5m",
                "update_frequency": "1m"
            }
        }, name="real_time_processing")

        performance_results = exp.run()

    return performance_results
```

## Advanced Monitoring Strategies

### 1. Predictive Monitoring

```python
def setup_predictive_monitoring():
    """Set up predictive monitoring to anticipate issues."""

    predictive_config = {
        "prediction_models": {
            "resource_prediction": {
                "model_type": "lstm",
                "prediction_horizon": "1h",
                "features": ["cpu_usage", "memory_usage", "network_io"],
                "target": "resource_exhaustion"
            },
            "performance_prediction": {
                "model_type": "random_forest",
                "prediction_horizon": "30m",
                "features": ["response_time", "throughput", "error_rate"],
                "target": "performance_degradation"
            },
            "failure_prediction": {
                "model_type": "survival_analysis",
                "prediction_horizon": "24h",
                "features": ["system_metrics", "error_patterns"],
                "target": "system_failure"
            }
        },
        "early_warning": {
            "confidence_threshold": 0.8,
            "warning_horizon": "30m",
            "mitigation_actions": ["scale_up", "restart", "rollback"]
        }
    }

    # Set up predictive monitoring
    with run.Experiment("predictive_monitoring") as exp:
        exp.add(predictive_config, name="predictive_config")

        # Add model training and updating
        exp.add({
            "model_management": {
                "retraining_schedule": "weekly",
                "performance_evaluation": "daily",
                "model_versioning": True
            }
        }, name="model_management")

        predictive_results = exp.run()

    return predictive_results
```

### 2. Distributed Monitoring

```python
def setup_distributed_monitoring():
    """Set up monitoring for distributed ML systems."""

    distributed_monitoring_config = {
        "monitoring_nodes": {
            "coordinator": {
                "role": "aggregation",
                "location": "central",
                "responsibilities": ["data_aggregation", "alert_coordination"]
            },
            "edge_nodes": [
                {
                    "location": "region_1",
                    "services": ["model_inference", "data_collection"],
                    "monitoring": ["local_metrics", "health_checks"]
                },
                {
                    "location": "region_2",
                    "services": ["model_inference", "data_collection"],
                    "monitoring": ["local_metrics", "health_checks"]
                }
            ]
        },
        "data_synchronization": {
            "sync_frequency": "1m",
            "data_compression": True,
            "redundancy": True
        },
        "fault_tolerance": {
            "node_failure_handling": "automatic",
            "data_replication": True,
            "service_discovery": True
        }
    }

    # Set up distributed monitoring
    with run.Experiment("distributed_monitoring") as exp:
        exp.add(distributed_monitoring_config, name="distributed_config")

        # Add coordination logic
        exp.add({
            "coordination": {
                "load_balancing": True,
                "failover": True,
                "consistency": "eventual"
            }
        }, name="coordination")

        distributed_results = exp.run()

    return distributed_results
```

### 3. Custom Metrics and Dashboards

```python
def setup_custom_monitoring():
    """Set up custom metrics and dashboards for specific use cases."""

    custom_metrics_config = {
        "business_metrics": {
            "user_engagement": {
                "calculation": "active_users / total_users",
                "threshold": 0.7,
                "alert": True
            },
            "revenue_impact": {
                "calculation": "revenue_with_model - revenue_without_model",
                "threshold": 1000,
                "alert": True
            },
            "model_roi": {
                "calculation": "revenue_increase / model_cost",
                "threshold": 5.0,
                "alert": False
            }
        },
        "technical_metrics": {
            "model_efficiency": {
                "calculation": "predictions_per_second / cpu_usage",
                "threshold": 100,
                "alert": True
            },
            "data_quality": {
                "calculation": "valid_predictions / total_predictions",
                "threshold": 0.95,
                "alert": True
            },
            "system_reliability": {
                "calculation": "uptime / total_time",
                "threshold": 0.99,
                "alert": True
            }
        },
        "custom_dashboards": {
            "executive_dashboard": {
                "metrics": ["revenue_impact", "user_engagement", "model_roi"],
                "refresh_rate": "5m",
                "visualization": "charts"
            },
            "technical_dashboard": {
                "metrics": ["model_efficiency", "data_quality", "system_reliability"],
                "refresh_rate": "1m",
                "visualization": "real_time"
            },
            "operational_dashboard": {
                "metrics": ["response_time", "error_rate", "throughput"],
                "refresh_rate": "30s",
                "visualization": "gauges"
            }
        }
    }

    # Set up custom monitoring
    with run.Experiment("custom_monitoring") as exp:
        exp.add(custom_metrics_config, name="custom_metrics")

        # Add dashboard configuration
        exp.add({
            "dashboard_config": {
                "auto_refresh": True,
                "export_capabilities": True,
                "sharing_permissions": True
            }
        }, name="dashboard_config")

        custom_results = exp.run()

    return custom_results
```

## Alerting and Incident Management

### 1. Multi-Level Alerting

```python
def setup_multi_level_alerting():
    """Set up multi-level alerting system."""

    alerting_config = {
        "alert_levels": {
            "critical": {
                "conditions": ["system_down", "data_loss", "security_breach"],
                "response_time": "immediate",
                "channels": ["pagerduty", "phone", "email"],
                "escalation": "automatic"
            },
            "high": {
                "conditions": ["performance_degradation", "high_error_rate"],
                "response_time": "5m",
                "channels": ["slack", "email"],
                "escalation": "manual"
            },
            "medium": {
                "conditions": ["trend_changes", "resource_usage_high"],
                "response_time": "15m",
                "channels": ["slack", "email"],
                "escalation": "none"
            },
            "low": {
                "conditions": ["informational", "maintenance_required"],
                "response_time": "1h",
                "channels": ["email"],
                "escalation": "none"
            }
        },
        "notification_channels": {
            "email": {
                "recipients": ["oncall@company.com", "ml-team@company.com"],
                "template": "alert_email_template"
            },
            "slack": {
                "channels": ["#ml-alerts", "#oncall"],
                "template": "alert_slack_template"
            },
            "pagerduty": {
                "service_id": "ml-monitoring-service",
                "urgency": "high"
            }
        }
    }

    # Set up alerting
    with run.Experiment("multi_level_alerting") as exp:
        exp.add(alerting_config, name="alerting_config")

        # Add incident management
        exp.add({
            "incident_management": {
                "auto_creation": True,
                "escalation_rules": True,
                "resolution_tracking": True
            }
        }, name="incident_management")

        alerting_results = exp.run()

    return alerting_results
```

### 2. Automated Response

```python
def setup_automated_response():
    """Set up automated response to common issues."""

    automated_response_config = {
        "response_actions": {
            "high_cpu_usage": {
                "triggers": ["cpu_usage > 80% for 5m"],
                "actions": ["scale_up", "restart_service"],
                "cooldown": "10m"
            },
            "high_error_rate": {
                "triggers": ["error_rate > 5% for 2m"],
                "actions": ["rollback_model", "restart_service"],
                "cooldown": "5m"
            },
            "high_latency": {
                "triggers": ["response_time > 200ms for 3m"],
                "actions": ["scale_up", "optimize_model"],
                "cooldown": "15m"
            },
            "model_drift": {
                "triggers": ["accuracy_drop > 2% for 1h"],
                "actions": ["switch_to_fallback", "schedule_retraining"],
                "cooldown": "1h"
            }
        },
        "safety_checks": {
            "max_scale_up": 5,
            "max_restarts": 3,
            "rollback_safety": True,
            "human_approval": ["model_rollback", "major_changes"]
        }
    }

    # Set up automated response
    with run.Experiment("automated_response") as exp:
        exp.add(automated_response_config, name="automated_response_config")

        # Add response monitoring
        exp.add({
            "response_monitoring": {
                "effectiveness_tracking": True,
                "response_time_measurement": True,
                "false_positive_tracking": True
            }
        }, name="response_monitoring")

        response_results = exp.run()

    return response_results
```

## Data Collection and Storage

### 1. Metrics Collection

```python
def setup_metrics_collection():
    """Set up comprehensive metrics collection."""

    metrics_collection_config = {
        "data_sources": {
            "model_metrics": {
                "source": "model_inference",
                "frequency": "real_time",
                "metrics": ["accuracy", "latency", "throughput"]
            },
            "system_metrics": {
                "source": "infrastructure",
                "frequency": "1m",
                "metrics": ["cpu", "memory", "disk", "network"]
            },
            "business_metrics": {
                "source": "application",
                "frequency": "5m",
                "metrics": ["user_engagement", "revenue", "conversion"]
            },
            "custom_metrics": {
                "source": "application",
                "frequency": "1m",
                "metrics": ["model_confidence", "feature_importance"]
            }
        },
        "data_processing": {
            "aggregation": {
                "time_windows": ["1m", "5m", "1h", "1d"],
                "functions": ["mean", "std", "min", "max", "count"]
            },
            "filtering": {
                "remove_outliers": True,
                "data_validation": True,
                "quality_checks": True
            },
            "enrichment": {
                "add_metadata": True,
                "add_context": True,
                "add_labels": True
            }
        },
        "storage": {
            "time_series_db": {
                "type": "influxdb",
                "retention": "90d",
                "compression": True
            },
            "data_warehouse": {
                "type": "bigquery",
                "retention": "1y",
                "partitioning": "daily"
            }
        }
    }

    # Set up metrics collection
    with run.Experiment("metrics_collection") as exp:
        exp.add(metrics_collection_config, name="metrics_collection_config")

        # Add data pipeline
        exp.add({
            "data_pipeline": {
                "stream_processing": True,
                "batch_processing": True,
                "data_quality_monitoring": True
            }
        }, name="data_pipeline")

        collection_results = exp.run()

    return collection_results
```

### 2. Logging and Tracing

```python
def setup_logging_and_tracing():
    """Set up comprehensive logging and tracing."""

    logging_config = {
        "log_levels": {
            "debug": {
                "enabled": True,
                "retention": "7d",
                "sampling": 0.1
            },
            "info": {
                "enabled": True,
                "retention": "30d",
                "sampling": 1.0
            },
            "warning": {
                "enabled": True,
                "retention": "90d",
                "sampling": 1.0
            },
            "error": {
                "enabled": True,
                "retention": "1y",
                "sampling": 1.0
            }
        },
        "structured_logging": {
            "format": "json",
            "fields": ["timestamp", "level", "service", "trace_id", "user_id"],
            "correlation": True
        },
        "distributed_tracing": {
            "enabled": True,
            "sampling_rate": 0.1,
            "trace_retention": "7d",
            "correlation": True
        }
    }

    # Set up logging and tracing
    with run.Experiment("logging_tracing") as exp:
        exp.add(logging_config, name="logging_config")

        # Add log analysis
        exp.add({
            "log_analysis": {
                "pattern_detection": True,
                "anomaly_detection": True,
                "correlation_analysis": True
            }
        }, name="log_analysis")

        logging_results = exp.run()

    return logging_results
```

## Best Practices

### 1. Monitoring Strategy

```python
def monitoring_best_practices():
    """Define monitoring best practices."""

    best_practices = {
        "metric_selection": {
            "golden_signals": ["latency", "traffic", "errors", "saturation"],
            "business_metrics": ["revenue", "user_satisfaction", "conversion"],
            "technical_metrics": ["cpu", "memory", "disk", "network"]
        },
        "alerting": {
            "alert_on_symptoms": True,
            "avoid_alerting_on_causes": True,
            "use_appropriate_thresholds": True,
            "implement_alert_fatigue_prevention": True
        },
        "dashboards": {
            "keep_it_simple": True,
            "focus_on_actionable_metrics": True,
            "use_appropriate_visualizations": True,
            "maintain_consistency": True
        },
        "documentation": {
            "document_metrics": True,
            "explain_thresholds": True,
            "maintain_runbooks": True,
            "update_regularly": True
        }
    }

    return best_practices
```

### 2. Performance Optimization

```python
def optimize_monitoring_performance():
    """Optimize monitoring system performance."""

    optimization_config = {
        "data_optimization": {
            "sampling": {
                "high_volume_metrics": 0.1,
                "medium_volume_metrics": 0.5,
                "low_volume_metrics": 1.0
            },
            "compression": {
                "time_series_data": True,
                "log_data": True,
                "trace_data": True
            },
            "aggregation": {
                "real_time": "1m",
                "short_term": "5m",
                "long_term": "1h"
            }
        },
        "system_optimization": {
            "resource_allocation": {
                "monitoring_overhead": "< 5%",
                "storage_efficiency": "optimized",
                "network_usage": "minimized"
            },
            "scalability": {
                "horizontal_scaling": True,
                "load_balancing": True,
                "auto_scaling": True
            }
        }
    }

    return optimization_config
```

## Summary

NeMo Run provides comprehensive monitoring capabilities that enable teams to:

- **Monitor Effectively**: Real-time metrics collection, performance tracking, and resource monitoring
- **Detect Issues Early**: Anomaly detection, predictive monitoring, and trend analysis
- **Respond Quickly**: Multi-level alerting, automated response, and incident management
- **Optimize Performance**: Cost monitoring, efficiency tracking, and resource optimization
- **Ensure Reliability**: Health checks, fault tolerance, and distributed monitoring

By leveraging these capabilities, organizations can maintain high-performing ML systems with minimal downtime and maximum efficiency.
