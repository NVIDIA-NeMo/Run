---
description: "Best practices for optimizing execution workflows for different environments and platforms."
categories: ["concepts-architecture"]
tags: ["best-practices", "execution", "performance", "optimization", "resource-management"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(execution-best-practices)=

# Execution Best Practices

This guide covers best practices for optimizing execution workflows for different environments and platforms, ensuring efficient resource usage and reliable execution.

## Performance Optimization

### Resource-Aware Execution

```python
import nemo_run as run
import psutil
import os
from typing import Dict, Any

class ResourceMonitor:
    """Monitor system resources during execution."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get current system resource information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

    @staticmethod
    def check_resource_availability() -> bool:
        """Check if system has sufficient resources for execution."""
        info = ResourceMonitor.get_system_info()

        # Check memory availability (require at least 20% free)
        if info["memory_percent"] > 80:
            return False

        # Check disk space (require at least 10% free)
        if info["disk_usage"] > 90:
            return False

        return True

def create_resource_aware_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment that checks resources before execution."""

    def resource_aware_training(model_config, training_config):
        """Training function that checks resources before starting."""
        if not ResourceMonitor.check_resource_availability():
            raise RuntimeError("Insufficient system resources for training")

        # Adjust batch size based on available memory
        system_info = ResourceMonitor.get_system_info()
        available_memory_gb = system_info["memory_available"] / (1024**3)

        # Adjust batch size: 1GB per batch size unit
        max_batch_size = max(1, int(available_memory_gb))
        adjusted_batch_size = min(training_config["batch_size"], max_batch_size)

        print(f"Adjusted batch size from {training_config['batch_size']} to {adjusted_batch_size}")
        training_config["batch_size"] = adjusted_batch_size

        # Proceed with training
        return train_model(model_config, training_config)

    return run.Experiment([
        run.Task(
            "resource_aware_training",
            run.Partial(resource_aware_training, model_config, training_config)
        )
    ])
```

### Parallel Execution Strategies

```python
import nemo_run as run
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from typing import List, Callable, Any

class ExecutionStrategy:
    """Different execution strategies for various scenarios."""

    @staticmethod
    def sequential_execution(tasks: List[Callable]) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        for task in tasks:
            results.append(task())
        return results

    @staticmethod
    def threaded_execution(tasks: List[Callable], max_workers: int = None) -> List[Any]:
        """Execute tasks using thread pool."""
        if max_workers is None:
            max_workers = min(len(tasks), multiprocessing.cpu_count())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = [future.result() for future in futures]
        return results

    @staticmethod
    def process_execution(tasks: List[Callable], max_workers: int = None) -> List[Any]:
        """Execute tasks using process pool for CPU-intensive work."""
        if max_workers is None:
            max_workers = min(len(tasks), multiprocessing.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = [future.result() for future in futures]
        return results

def create_parallel_experiment(
    model_configs: List[run.Config],
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment that runs multiple model configurations in parallel."""

    def train_single_model(model_config, training_config, model_id):
        """Train a single model with given configuration."""
        print(f"Training model {model_id}")
        return train_model(model_config, training_config)

    # Create tasks for each model configuration
    tasks = []
    for i, model_config in enumerate(model_configs):
        task = run.Task(
            f"model_training_{i}",
            run.Partial(train_single_model, model_config, training_config, i)
        )
        tasks.append(task)

    return run.Experiment(tasks)
```

## Memory Management

### Efficient Memory Usage

```python
import nemo_run as run
import gc
import torch
from typing import Optional

class MemoryManager:
    """Manage memory usage during execution."""

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def clear_cpu_memory():
        """Clear CPU memory."""
        gc.collect()

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            "percent": process.memory_percent()
        }

def create_memory_efficient_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment with memory management."""

    def memory_efficient_training(model_config, training_config):
        """Training function with memory management."""
        try:
            # Initial memory check
            initial_memory = MemoryManager.get_memory_usage()
            print(f"Initial memory usage: {initial_memory['rss_mb']:.2f} MB")

            # Training
            model = train_model(model_config, training_config)

            # Clear memory after training
            MemoryManager.clear_gpu_memory()
            MemoryManager.clear_cpu_memory()

            # Final memory check
            final_memory = MemoryManager.get_memory_usage()
            print(f"Final memory usage: {final_memory['rss_mb']:.2f} MB")

            return model

        except Exception as e:
            # Ensure memory is cleared even on error
            MemoryManager.clear_gpu_memory()
            MemoryManager.clear_cpu_memory()
            raise e

    return run.Experiment([
        run.Task(
            "memory_efficient_training",
            run.Partial(memory_efficient_training, model_config, training_config)
        )
    ])
```

### Batch Processing for Large Datasets

```python
import nemo_run as run
from typing import Iterator, List, Any

def create_batch_processor(
    data_iterator: Iterator[Any],
    batch_size: int,
    process_function: Callable
) -> Iterator[Any]:
    """Process data in batches to manage memory."""

    batch = []
    for item in data_iterator:
        batch.append(item)

        if len(batch) >= batch_size:
            yield process_function(batch)
            batch = []

    # Process remaining items
    if batch:
        yield process_function(batch)

def create_batch_processing_experiment(
    data_config: run.Config,
    model_config: run.Config,
    batch_size: int = 1000
) -> run.Experiment:
    """Create an experiment that processes data in batches."""

    def batch_process_data(data_config, model_config, batch_size):
        """Process data in batches to avoid memory issues."""
        data_iterator = data_config.build()

        def process_batch(batch):
            """Process a single batch of data."""
            # Process batch
            results = []
            for item in batch:
                result = process_item(item, model_config)
                results.append(result)
            return results

        # Process in batches
        batch_processor = create_batch_processor(
            data_iterator, batch_size, process_batch
        )

        all_results = []
        for batch_result in batch_processor:
            all_results.extend(batch_result)

            # Clear memory after each batch
            MemoryManager.clear_cpu_memory()

        return all_results

    return run.Experiment([
        run.Task(
            "batch_processing",
            run.Partial(batch_process_data, data_config, model_config, batch_size)
        )
    ])
```

## Error Handling and Recovery

### Robust Execution with Retries

```python
import nemo_run as run
import time
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

class ExecutionRetry:
    """Handle execution retries with exponential backoff."""

    @staticmethod
    def execute_with_retry(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ) -> Any:
        """Execute function with retry logic."""

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func()

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
                    raise last_exception

def create_robust_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any],
    max_retries: int = 3
) -> run.Experiment:
    """Create an experiment with robust error handling."""

    def robust_training(model_config, training_config):
        """Training function with retry logic."""

        def training_function():
            """The actual training function."""
            return train_model(model_config, training_config)

        return ExecutionRetry.execute_with_retry(
            training_function,
            max_retries=max_retries
        )

    return run.Experiment([
        run.Task(
            "robust_training",
            run.Partial(robust_training, model_config, training_config)
        )
    ])
```

### Graceful Degradation

```python
import nemo_run as run
from typing import Optional, Dict, Any

class GracefulDegradation:
    """Handle graceful degradation when resources are limited."""

    @staticmethod
    def adaptive_batch_size(
        initial_batch_size: int,
        memory_threshold: float = 0.8
    ) -> int:
        """Adapt batch size based on available memory."""
        import psutil

        memory_percent = psutil.virtual_memory().percent / 100

        if memory_percent > memory_threshold:
            # Reduce batch size when memory usage is high
            reduction_factor = 1 - (memory_percent - memory_threshold)
            return max(1, int(initial_batch_size * reduction_factor))

        return initial_batch_size

    @staticmethod
    def adaptive_epochs(
        initial_epochs: int,
        time_constraint: Optional[float] = None
    ) -> int:
        """Adapt number of epochs based on time constraints."""
        if time_constraint is None:
            return initial_epochs

        # Estimate time per epoch and adjust
        estimated_time_per_epoch = 60  # seconds (this would be estimated)
        max_epochs = int(time_constraint / estimated_time_per_epoch)

        return min(initial_epochs, max_epochs)

def create_adaptive_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment that adapts to resource constraints."""

    def adaptive_training(model_config, training_config):
        """Training function that adapts to available resources."""

        # Adapt batch size
        adaptive_batch_size = GracefulDegradation.adaptive_batch_size(
            training_config["batch_size"]
        )
        training_config["batch_size"] = adaptive_batch_size

        # Adapt epochs if time constraint is specified
        if "time_constraint" in training_config:
            adaptive_epochs = GracefulDegradation.adaptive_epochs(
                training_config["epochs"],
                training_config["time_constraint"]
            )
            training_config["epochs"] = adaptive_epochs

        print(f"Adapted batch size: {adaptive_batch_size}")
        print(f"Adapted epochs: {training_config['epochs']}")

        return train_model(model_config, training_config)

    return run.Experiment([
        run.Task(
            "adaptive_training",
            run.Partial(adaptive_training, model_config, training_config)
        )
    ])
```

## Execution Monitoring

### Real-time Monitoring

```python
import nemo_run as run
import time
import threading
from typing import Dict, Any, Optional

class ExecutionMonitor:
    """Monitor execution progress and performance."""

    def __init__(self):
        self.start_time = None
        self.metrics = {}
        self.is_monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start monitoring execution."""
        self.start_time = time.time()
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring execution."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Monitoring loop that runs in background."""
        while self.is_monitoring:
            # Collect metrics
            self.metrics.update(self._collect_metrics())
            time.sleep(1)  # Update every second

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current execution metrics."""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

def create_monitored_experiment(
    model_config: run.Config,
    training_config: Dict[str, Any]
) -> run.Experiment:
    """Create an experiment with execution monitoring."""

    def monitored_training(model_config, training_config):
        """Training function with monitoring."""
        monitor = ExecutionMonitor()

        try:
            # Start monitoring
            monitor.start_monitoring()

            # Training
            result = train_model(model_config, training_config)

            # Get final metrics
            final_metrics = monitor.get_metrics()
            print(f"Final metrics: {final_metrics}")

            return result

        finally:
            # Stop monitoring
            monitor.stop_monitoring()

    return run.Experiment([
        run.Task(
            "monitored_training",
            run.Partial(monitored_training, model_config, training_config)
        )
    ])
```

## Best Practices Summary

### Do's

- ✅ **Monitor resources** before and during execution
- ✅ **Use appropriate execution strategies** (sequential, threaded, process)
- ✅ **Implement memory management** for large datasets
- ✅ **Handle errors gracefully** with retry logic
- ✅ **Adapt to resource constraints** with graceful degradation
- ✅ **Monitor execution progress** in real-time
- ✅ **Clear memory** after processing batches
- ✅ **Use batch processing** for large datasets

### Don'ts

- ❌ **Ignore resource constraints** during execution
- ❌ **Use inappropriate execution strategies** for the task
- ❌ **Let memory leaks** accumulate during execution
- ❌ **Fail silently** without proper error handling
- ❌ **Use fixed parameters** without considering resource availability
- ❌ **Skip monitoring** in production environments
- ❌ **Process large datasets** without batching

## Next Steps

- Review [Configuration Best Practices](configuration-best-practices)
- Learn about [Management Best Practices](management-best-practices)
- Explore [Team Collaboration](team-collaboration) guidelines
- Check [Troubleshooting](../reference/troubleshooting) for execution issues
