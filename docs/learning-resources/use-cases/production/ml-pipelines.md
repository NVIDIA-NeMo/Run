---
description: "Build robust, scalable ML pipelines for production environments using NeMo Run"
categories: ["use-cases", "production"]
tags: ["ml-pipelines", "production", "orchestration", "scalability"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "advanced"
content_type: "use-case"
modality: "text-only"
---

# ML Pipelines

Build robust, scalable ML pipelines for production environments with NeMo Run's end-to-end pipeline orchestration, fault tolerance, and monitoring capabilities.

## Overview

ML pipelines in NeMo Run provide comprehensive tools for building, deploying, and managing production-ready machine learning workflows with automated orchestration, monitoring, and scaling capabilities.

## Key Features

### End-to-End Pipeline Orchestration
- Complete workflow automation
- Dependency management
- Parallel execution capabilities
- Resource optimization

### Fault Tolerance and Recovery
- Automatic error handling
- Retry mechanisms
- State persistence
- Rollback capabilities

### Monitoring and Alerting
- Real-time pipeline monitoring
- Performance metrics tracking
- Automated alerting systems
- Health check mechanisms

### Automated Deployment
- Continuous integration/deployment
- Environment management
- Configuration management
- Version control integration

## Implementation

### Pipeline Architecture

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import time

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineStep:
    """Individual step in a production ML pipeline."""

    name: str
    function: Callable
    dependencies: List[str]
    resources: Dict[str, Any]
    timeout: int
    retries: int
    status: PipelineStatus = PipelineStatus.PENDING

    def execute(self, context: Dict[str, Any]):
        """Execute the pipeline step."""

        self.status = PipelineStatus.RUNNING
        start_time = time.time()

        try:
            # Execute with timeout and retries
            result = self._execute_with_retries(context)

            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Step {self.name} exceeded timeout of {self.timeout}s")

            self.status = PipelineStatus.COMPLETED
            return result

        except Exception as e:
            self.status = PipelineStatus.FAILED
            raise e

    def _execute_with_retries(self, context: Dict[str, Any]):
        """Execute step with retry logic."""

        last_exception = None

        for attempt in range(self.retries + 1):
            try:
                return self.function(context)
            except Exception as e:
                last_exception = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise last_exception

class ProductionPipeline:
    """Production-ready ML pipeline with monitoring and fault tolerance."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: Dict[str, PipelineStep] = {}
        self.status = PipelineStatus.PENDING
        self.context = {}
        self.metrics = {}
        self.start_time = None
        self.end_time = None

    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline."""

        self.steps[step.name] = step
        return self

    def execute(self, initial_context: Dict[str, Any] = None):
        """Execute the complete pipeline."""

        self.status = PipelineStatus.RUNNING
        self.start_time = time.time()
        self.context = initial_context or {}

        try:
            # Validate dependencies
            self._validate_dependencies()

            # Execute steps in dependency order
            execution_order = self._determine_execution_order()

            for step_name in execution_order:
                step = self.steps[step_name]

                # Execute step
                result = step.execute(self.context)

                # Store result in context
                self.context[step_name] = result

                # Update metrics
                self._update_metrics(step_name, result)

                # Check pipeline health
                self._health_check()

            self.status = PipelineStatus.COMPLETED
            self.end_time = time.time()

            return self.context

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.end_time = time.time()
            self._handle_failure(e)
            raise e

    def _validate_dependencies(self):
        """Validate that all dependencies exist."""

        for step_name, step in self.steps.items():
            for dependency in step.dependencies:
                if dependency not in self.steps:
                    raise ValueError(f"Step {step_name} depends on {dependency}, but it doesn't exist")

    def _determine_execution_order(self):
        """Determine the order of step execution based on dependencies."""

        # Topological sort
        in_degree = {step_name: 0 for step_name in self.steps}
        graph = {step_name: [] for step_name in self.steps}

        # Build dependency graph
        for step_name, step in self.steps.items():
            for dependency in step.dependencies:
                graph[dependency].append(step_name)
                in_degree[step_name] += 1

        # Topological sort
        execution_order = []
        queue = [step_name for step_name, degree in in_degree.items() if degree == 0]

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(execution_order) != len(self.steps):
            raise ValueError("Circular dependency detected in pipeline")

        return execution_order

    def _update_metrics(self, step_name: str, result: Any):
        """Update pipeline metrics."""

        self.metrics[step_name] = {
            "status": self.steps[step_name].status.value,
            "result": result,
            "timestamp": time.time()
        }

    def _health_check(self):
        """Perform health check on pipeline."""

        # Check resource usage
        resource_usage = self._check_resource_usage()
        if resource_usage > 0.9:  # 90% threshold
            run.log_warning(f"High resource usage detected: {resource_usage}")

        # Check step failures
        failed_steps = [name for name, step in self.steps.items()
                       if step.status == PipelineStatus.FAILED]
        if failed_steps:
            run.log_error(f"Failed steps: {failed_steps}")

    def _handle_failure(self, error: Exception):
        """Handle pipeline failure."""

        # Log failure
        run.log_error(f"Pipeline {self.name} failed: {error}")

        # Send alert
        self._send_alert(f"Pipeline {self.name} failed", str(error))

        # Attempt recovery
        self._attempt_recovery()

    def _attempt_recovery(self):
        """Attempt to recover from failure."""

        # Check if any steps can be retried
        for step_name, step in self.steps.items():
            if step.status == PipelineStatus.FAILED and step.retries > 0:
                run.log_info(f"Retrying step {step_name}")
                try:
                    step.execute(self.context)
                    step.status = PipelineStatus.COMPLETED
                except Exception as e:
                    run.log_error(f"Recovery failed for step {step_name}: {e}")

    def _check_resource_usage(self):
        """Check current resource usage."""

        # This would integrate with actual resource monitoring
        return 0.5  # Placeholder

    def _send_alert(self, title: str, message: str):
        """Send alert notification."""

        alert = {
            "title": title,
            "message": message,
            "pipeline": self.name,
            "timestamp": time.time(),
            "severity": "high"
        }

        run.send_alert(alert)

    def get_status(self):
        """Get current pipeline status."""

        return {
            "name": self.name,
            "status": self.status.value,
            "steps": {name: step.status.value for name, step in self.steps.items()},
            "metrics": self.metrics,
            "duration": (self.end_time or time.time()) - (self.start_time or 0),
            "context_keys": list(self.context.keys())
        }

# Initialize production pipeline
pipeline = ProductionPipeline("recommendation_pipeline", "Production recommendation system pipeline")
```

### Data Processing Pipeline

```python
def data_processing_pipeline():
    """Example data processing pipeline for production."""

    # Define pipeline steps
    def load_data(context):
        """Load and validate input data."""
        run.log_info("Loading data...")
        # Simulate data loading
        data = {"users": 1000, "items": 5000, "interactions": 50000}
        run.log_info(f"Loaded {data['interactions']} interactions")
        return data

    def clean_data(context):
        """Clean and preprocess data."""
        run.log_info("Cleaning data...")
        data = context["load_data"]
        # Simulate data cleaning
        cleaned_data = {**data, "cleaned_interactions": 48000}
        run.log_info(f"Cleaned data: {cleaned_data['cleaned_interactions']} interactions")
        return cleaned_data

    def feature_engineering(context):
        """Extract and engineer features."""
        run.log_info("Engineering features...")
        cleaned_data = context["clean_data"]
        # Simulate feature engineering
        features = {
            "user_features": 1000,
            "item_features": 5000,
            "interaction_features": 48000
        }
        run.log_info(f"Engineered {sum(features.values())} features")
        return features

    def train_model(context):
        """Train the recommendation model."""
        run.log_info("Training model...")
        features = context["feature_engineering"]
        # Simulate model training
        model = {
            "type": "collaborative_filtering",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78
        }
        run.log_info(f"Model trained with accuracy: {model['accuracy']}")
        return model

    def evaluate_model(context):
        """Evaluate model performance."""
        run.log_info("Evaluating model...")
        model = context["train_model"]
        # Simulate evaluation
        evaluation = {
            "test_accuracy": 0.83,
            "test_precision": 0.80,
            "test_recall": 0.76,
            "overfitting": False
        }
        run.log_info(f"Model evaluation complete: {evaluation['test_accuracy']} accuracy")
        return evaluation

    def deploy_model(context):
        """Deploy model to production."""
        run.log_info("Deploying model...")
        model = context["train_model"]
        evaluation = context["evaluate_model"]

        # Check if model meets deployment criteria
        if evaluation["test_accuracy"] >= 0.80 and not evaluation["overfitting"]:
            deployment = {
                "status": "deployed",
                "version": "v1.0.0",
                "endpoint": "https://api.example.com/recommendations",
                "timestamp": time.time()
            }
            run.log_info("Model deployed successfully")
        else:
            deployment = {
                "status": "rejected",
                "reason": "Model performance below threshold"
            }
            run.log_warning("Model deployment rejected")

        return deployment

    # Create pipeline
    pipeline = ProductionPipeline("recommendation_pipeline", "Production recommendation system")

    # Add steps with dependencies
    pipeline.add_step(PipelineStep(
        name="load_data",
        function=load_data,
        dependencies=[],
        resources={"cpu": 2, "memory": "4GB"},
        timeout=300,
        retries=2
    ))

    pipeline.add_step(PipelineStep(
        name="clean_data",
        function=clean_data,
        dependencies=["load_data"],
        resources={"cpu": 4, "memory": "8GB"},
        timeout=600,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="feature_engineering",
        function=feature_engineering,
        dependencies=["clean_data"],
        resources={"cpu": 8, "memory": "16GB"},
        timeout=900,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="train_model",
        function=train_model,
        dependencies=["feature_engineering"],
        resources={"cpu": 16, "memory": "32GB", "gpu": 1},
        timeout=1800,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="evaluate_model",
        function=evaluate_model,
        dependencies=["train_model"],
        resources={"cpu": 4, "memory": "8GB"},
        timeout=300,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="deploy_model",
        function=deploy_model,
        dependencies=["evaluate_model"],
        resources={"cpu": 2, "memory": "4GB"},
        timeout=300,
        retries=2
    ))

    return pipeline

# Create and execute pipeline
recommendation_pipeline = data_processing_pipeline()
results = recommendation_pipeline.execute()
```

### Monitoring and Alerting

```python
class PipelineMonitor:
    """Monitor production pipelines and send alerts."""

    def __init__(self, pipeline: ProductionPipeline):
        self.pipeline = pipeline
        self.monitoring_config = {
            "health_check_interval": 30,  # seconds
            "alert_thresholds": {
                "step_failure_rate": 0.1,
                "pipeline_duration": 3600,  # 1 hour
                "resource_usage": 0.9
            }
        }

    def start_monitoring(self):
        """Start monitoring the pipeline."""

        run.log_info(f"Starting monitoring for pipeline {self.pipeline.name}")

        # Monitor pipeline execution
        while self.pipeline.status == PipelineStatus.RUNNING:
            self._check_pipeline_health()
            time.sleep(self.monitoring_config["health_check_interval"])

        # Final health check
        self._check_pipeline_health()

    def _check_pipeline_health(self):
        """Check pipeline health and send alerts if needed."""

        status = self.pipeline.get_status()

        # Check step failure rate
        failed_steps = [step for step in status["steps"].values()
                       if step == PipelineStatus.FAILED.value]
        failure_rate = len(failed_steps) / len(status["steps"])

        if failure_rate > self.monitoring_config["alert_thresholds"]["step_failure_rate"]:
            self._send_alert(
                "High step failure rate",
                f"Pipeline {self.pipeline.name} has {failure_rate:.2%} failure rate"
            )

        # Check pipeline duration
        if status["duration"] > self.monitoring_config["alert_thresholds"]["pipeline_duration"]:
            self._send_alert(
                "Pipeline taking too long",
                f"Pipeline {self.pipeline.name} has been running for {status['duration']:.0f} seconds"
            )

        # Check resource usage
        resource_usage = self._check_resource_usage()
        if resource_usage > self.monitoring_config["alert_thresholds"]["resource_usage"]:
            self._send_alert(
                "High resource usage",
                f"Pipeline {self.pipeline.name} using {resource_usage:.1%} of resources"
            )

    def _check_resource_usage(self):
        """Check current resource usage."""

        # This would integrate with actual resource monitoring
        return 0.6  # Placeholder

    def _send_alert(self, title: str, message: str):
        """Send alert notification."""

        alert = {
            "title": title,
            "message": message,
            "pipeline": self.pipeline.name,
            "timestamp": time.time(),
            "severity": "warning"
        }

        run.send_alert(alert)
        run.log_warning(f"Alert sent: {title} - {message}")

# Initialize pipeline monitor
pipeline_monitor = PipelineMonitor(recommendation_pipeline)
```

## Use Cases

### Recommendation System Pipeline

**Scenario**: Production recommendation system with real-time updates

**Implementation**:
```python
def recommendation_pipeline():
    """Production recommendation system pipeline."""

    def collect_user_data(context):
        """Collect user interaction data."""
        # Collect real-time user interactions
        interactions = run.collect_user_interactions()
        return {"interactions": interactions}

    def preprocess_data(context):
        """Preprocess user interaction data."""
        data = context["collect_user_data"]
        # Clean and normalize data
        processed_data = run.preprocess_interactions(data["interactions"])
        return processed_data

    def update_model(context):
        """Update recommendation model with new data."""
        processed_data = context["preprocess_data"]
        # Incrementally update model
        updated_model = run.update_recommendation_model(processed_data)
        return updated_model

    def deploy_update(context):
        """Deploy updated model to production."""
        model = context["update_model"]
        # Deploy with zero downtime
        deployment = run.deploy_model_with_rollback(model)
        return deployment

    # Create pipeline
    pipeline = ProductionPipeline("recommendation_update", "Real-time recommendation updates")

    # Add steps
    pipeline.add_step(PipelineStep(
        name="collect_user_data",
        function=collect_user_data,
        dependencies=[],
        resources={"cpu": 2, "memory": "4GB"},
        timeout=300,
        retries=3
    ))

    pipeline.add_step(PipelineStep(
        name="preprocess_data",
        function=preprocess_data,
        dependencies=["collect_user_data"],
        resources={"cpu": 4, "memory": "8GB"},
        timeout=600,
        retries=2
    ))

    pipeline.add_step(PipelineStep(
        name="update_model",
        function=update_model,
        dependencies=["preprocess_data"],
        resources={"cpu": 8, "memory": "16GB", "gpu": 1},
        timeout=1800,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="deploy_update",
        function=deploy_update,
        dependencies=["update_model"],
        resources={"cpu": 2, "memory": "4GB"},
        timeout=300,
        retries=2
    ))

    return pipeline

# Execute recommendation pipeline
rec_pipeline = recommendation_pipeline()
monitor = PipelineMonitor(rec_pipeline)
monitor.start_monitoring()
results = rec_pipeline.execute()
```

### Fraud Detection Pipeline

**Scenario**: Real-time fraud detection system

**Implementation**:
```python
def fraud_detection_pipeline():
    """Real-time fraud detection pipeline."""

    def ingest_transactions(context):
        """Ingest real-time transaction data."""
        transactions = run.ingest_transaction_stream()
        return {"transactions": transactions}

    def extract_features(context):
        """Extract fraud detection features."""
        data = context["ingest_transactions"]
        features = run.extract_fraud_features(data["transactions"])
        return features

    def detect_fraud(context):
        """Detect fraudulent transactions."""
        features = context["extract_features"]
        predictions = run.detect_fraud(features)
        return predictions

    def take_action(context):
        """Take action on detected fraud."""
        predictions = context["detect_fraud"]
        actions = run.take_fraud_action(predictions)
        return actions

    # Create pipeline
    pipeline = ProductionPipeline("fraud_detection", "Real-time fraud detection")

    # Add steps with strict timing requirements
    pipeline.add_step(PipelineStep(
        name="ingest_transactions",
        function=ingest_transactions,
        dependencies=[],
        resources={"cpu": 4, "memory": "8GB"},
        timeout=60,  # Strict timeout for real-time
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="extract_features",
        function=extract_features,
        dependencies=["ingest_transactions"],
        resources={"cpu": 8, "memory": "16GB"},
        timeout=120,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="detect_fraud",
        function=detect_fraud,
        dependencies=["extract_features"],
        resources={"cpu": 16, "memory": "32GB", "gpu": 2},
        timeout=180,
        retries=1
    ))

    pipeline.add_step(PipelineStep(
        name="take_action",
        function=take_action,
        dependencies=["detect_fraud"],
        resources={"cpu": 4, "memory": "8GB"},
        timeout=60,
        retries=2
    ))

    return pipeline

# Execute fraud detection pipeline
fraud_pipeline = fraud_detection_pipeline()
results = fraud_pipeline.execute()
```

## Best Practices

### 1. Pipeline Design
- Design for fault tolerance and recovery
- Implement proper error handling
- Use appropriate timeouts and retries
- Plan for scalability and performance

### 2. Monitoring and Alerting
- Implement comprehensive monitoring
- Set up automated alerting
- Track performance metrics
- Monitor resource usage

### 3. Deployment
- Use continuous integration/deployment
- Implement blue-green deployments
- Maintain version control
- Test thoroughly before production

### 4. Security
- Implement proper access controls
- Secure data handling
- Monitor for security threats
- Regular security audits

## Success Metrics

### Pipeline Performance
- **Execution time**: Time to complete pipeline
- **Success rate**: Percentage of successful runs
- **Resource efficiency**: Resource usage optimization
- **Error rate**: Frequency of pipeline failures

### Production Metrics
- **Uptime**: Pipeline availability
- **Throughput**: Number of pipelines executed
- **Latency**: Time from trigger to completion
- **Scalability**: Ability to handle increased load

### Quality Metrics
- **Data quality**: Accuracy of processed data
- **Model performance**: Quality of ML models
- **Monitoring coverage**: Completeness of monitoring
- **Alert accuracy**: Precision of alerts

## Next Steps

- Explore **[Model Deployment](model-deployment.md)** for deployment strategies
- Learn about **[Ray Integration](../../../guides/ray.md)** for distributed training
- Review **[Guides](../../../guides/index.md)** for optimization strategies
- Consult **[References](../../../references/index.md)** for detailed API documentation
