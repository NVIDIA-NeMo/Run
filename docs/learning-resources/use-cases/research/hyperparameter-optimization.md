---
description: "Automate hyperparameter search with advanced optimization strategies using NeMo Run"
categories: ["use-cases", "research"]
tags: ["hyperparameter-optimization", "bayesian-optimization", "automated-ml", "research", "optimization"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "use-case"
modality: "text-only"
---

# Hyperparameter Optimization

Automate hyperparameter search with advanced optimization strategies using NeMo Run.

## Overview

Hyperparameter optimization is a critical component of ML research and development. NeMo Run provides powerful tools for automating this process with advanced optimization strategies, including Bayesian optimization, multi-objective optimization, and early stopping strategies.

## Key Features

### Multi-Objective Optimization

- Pareto frontier analysis
- Custom objective functions

### Bayesian Optimization

- Gaussian process-based optimization
- Uncertainty quantification
- Efficient exploration-exploitation

### Early Stopping Strategies

- Performance-based termination
- Time-budget management

## Use Case Scenarios
Concrete examples showing how to apply these strategies.

### Scenario 1: Neural Network Architecture Search

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, List
import optuna

@dataclass
class ArchitectureConfig:
    """Configuration for neural network architecture search."""
    num_layers: int
    hidden_sizes: List[int]
    activation: str
    dropout_rate: float
    learning_rate: float
    batch_size: int

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for architecture search."""

    # Define hyperparameter search space
    config = ArchitectureConfig(
        num_layers=trial.suggest_int("num_layers", 1, 5),
        hidden_sizes=[trial.suggest_int(f"hidden_size_{i}", 32, 512)
                     for i in range(trial.suggest_int("num_layers", 1, 5))],
        activation=trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"]),
        dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    )

    # Execute experiment with NeMo Run
    with run.Experiment(f"architecture_search_{trial.number}") as exp:
        exp.add(config, name="architecture_trial")
        results = exp.run()

        # Return validation accuracy as objective
        return results["validation_accuracy"]

def run_architecture_search():
    """Run neural network architecture search."""

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize with NeMo Run integration
    study.optimize(objective, n_trials=100, timeout=3600)

    # Get best configuration
    best_config = study.best_params
    best_value = study.best_value

    print(f"Best validation accuracy: {best_value:.4f}")
    print(f"Best configuration: {best_config}")

    return best_config, best_value
```

### Scenario 2: Multi-Objective Optimization

```python
@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    model_type: str
    hyperparameters: Dict[str, Any]
    objectives: List[str]

def multi_objective_function(trial: optuna.Trial) -> Dict[str, float]:
    """Multi-objective optimization function."""

    config = MultiObjectiveConfig(
        model_type=trial.suggest_categorical("model_type", ["transformer", "lstm", "cnn"]),
        hyperparameters={
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "hidden_size": trial.suggest_int("hidden_size", 64, 512)
        },
        objectives=["accuracy", "latency", "memory_usage"]
    )

    # Execute experiment
    with run.Experiment(f"multi_objective_{trial.number}") as exp:
        exp.add(config, name="multi_objective_trial")
        results = exp.run()

        # Return multiple objectives
        return {
            "accuracy": results["accuracy"],
            "latency": results["inference_latency"],
            "memory_usage": results["memory_usage"]
        }

def run_multi_objective_optimization():
    """Run multi-objective hyperparameter optimization."""

    # Create multi-objective study
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )

    # Optimize
    study.optimize(multi_objective_function, n_trials=50)

    # Analyze Pareto frontier
    pareto_front = study.best_trials

    print(f"Pareto frontier size: {len(pareto_front)}")
    for i, trial in enumerate(pareto_front):
        print(f"Trial {i}: {trial.values}")

    return pareto_front
```

### Scenario 3: Bayesian Optimization with Custom Acquisition

```python
class CustomAcquisitionFunction:
    """Custom acquisition function for Bayesian optimization."""

    def __init__(self, exploration_weight: float = 0.1):
        self.exploration_weight = exploration_weight

    def __call__(self, mean: float, std: float) -> float:
        """Custom acquisition function combining EI and exploration."""
        # Expected Improvement
        ei = max(0, mean - self.best_observed)

        # Exploration term
        exploration = std * self.exploration_weight

        return ei + exploration

def bayesian_optimization_with_custom_acquisition():
    """Run Bayesian optimization with custom acquisition function."""

    # Create study with custom sampler
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    # Run optimization
    study.optimize(objective, n_trials=100)

    return study
```

## Advanced Optimization Strategies
Scale beyond basics with evolutionary, bandit, and fidelity methods.

### 1. Population-Based Training

```python
def population_based_training():
    """Implement population-based training for hyperparameter optimization."""

    population_size = 10
    generations = 20

    # Initialize population
    population = []
    for i in range(population_size):
        config = generate_random_config()
        population.append(config)

    # Evolution loop
    for generation in range(generations):
        # Evaluate population
        fitness_scores = []
        for config in population:
            with run.Experiment(f"pbt_gen_{generation}_config_{len(fitness_scores)}") as exp:
                exp.add(config, name="pbt_trial")
                results = exp.run()
                fitness_scores.append(results["fitness"])

        # Selection and crossover
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return population
```

### 2. Hyperband Optimization

```python
def hyperband_optimization():
    """Implement Hyperband algorithm for resource-efficient optimization."""

    max_iter = 81  # Maximum iterations per configuration
    eta = 3        # Successive halving factor

    # Calculate number of brackets
    s_max = int(np.log(max_iter) / np.log(eta))
    B = (s_max + 1) * max_iter

    best_config = None
    best_score = float('-inf')

    for s in range(s_max, -1, -1):
        # Number of configurations in this bracket
        n = int(np.ceil(B / max_iter / (s + 1) * eta ** s))

        # Number of iterations per configuration
        r = max_iter * eta ** (-s)

        # Generate and evaluate configurations
        configs = [generate_random_config() for _ in range(n)]

        for i, config in enumerate(configs):
            with run.Experiment(f"hyperband_s{s}_i{i}") as exp:
                exp.add(config, name="hyperband_trial")
                results = exp.run(max_iterations=int(r))

                if results["score"] > best_score:
                    best_score = results["score"]
                    best_config = config

    return best_config, best_score
```

### 3. Multi-Fidelity Optimization

```python
def multi_fidelity_optimization():
    """Implement multi-fidelity optimization for efficient search."""

    # Define fidelity levels
    fidelity_levels = [0.1, 0.3, 0.5, 0.7, 1.0]  # Fraction of full training

    def low_fidelity_objective(trial: optuna.Trial, fidelity: float) -> float:
        """Objective function with fidelity control."""

        config = generate_config_from_trial(trial)

        with run.Experiment(f"multi_fidelity_{trial.number}") as exp:
            exp.add(config, name="multi_fidelity_trial")

            # Run with reduced fidelity
            results = exp.run(
                max_epochs=int(100 * fidelity),
                data_fraction=fidelity
            )

            return results["validation_accuracy"]

    # Create study
    study = optuna.create_study(direction="maximize")

    # Multi-fidelity optimization
    for fidelity in fidelity_levels:
        study.optimize(
            lambda trial: low_fidelity_objective(trial, fidelity),
            n_trials=20
        )

    return study
```

## Integration with NeMo Run Features
Leverage resource awareness and distributed execution.

### 1. Resource-Aware Optimization

```python
def resource_aware_optimization():
    """Optimize hyperparameters with resource constraints."""

    # Define resource constraints
    resource_constraints = {
        "max_memory": "32GB",
        "max_gpu_memory": "24GB",
        "max_time": "2h",
        "max_cost": "$50"
    }

    def constrained_objective(trial: optuna.Trial) -> float:
        """Objective function with resource monitoring."""

        config = generate_config_from_trial(trial)

        with run.Experiment(f"constrained_{trial.number}") as exp:
            exp.add(config, name="constrained_trial")

            # Monitor resources during execution
            results = exp.run(
                resource_monitoring=True,
                max_time=7200,  # 2 hours
                memory_limit="32GB"
            )

            # Check resource constraints
            if results["memory_usage"] > 32e9:  # 32GB
                return float('-inf')  # Penalty for constraint violation

            return results["validation_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(constrained_objective, n_trials=100)

    return study
```

### 2. Distributed Optimization

```python
def distributed_hyperparameter_optimization():
    """Run distributed hyperparameter optimization across multiple nodes."""

    # Configure distributed execution
    distributed_config = {
        "backend": "ray",
        "num_workers": 4,
        "resources_per_worker": {"CPU": 2, "GPU": 1}
    }

    def distributed_objective(trial: optuna.Trial) -> float:
        """Distributed objective function."""

        config = generate_config_from_trial(trial)

        with run.Experiment(f"distributed_{trial.number}") as exp:
            exp.add(config, name="distributed_trial")

            # Run on distributed cluster
            results = exp.run(
                backend="ray",
                cluster_config=distributed_config
            )

            return results["validation_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(distributed_objective, n_trials=200)

    return study
```

## Best Practices
Design effective search spaces and stopping rules.

### 1. Search Space Design

```python
def design_effective_search_space():
    """Design effective hyperparameter search spaces."""

    # Define search spaces based on domain knowledge
    search_spaces = {
        "learning_rate": {
            "type": "log_uniform",
            "min": 1e-5,
            "max": 1e-1
        },
        "batch_size": {
            "type": "categorical",
            "values": [16, 32, 64, 128, 256]
        },
        "architecture": {
            "type": "categorical",
            "values": ["resnet", "vgg", "inception", "efficientnet"]
        },
        "optimizer": {
            "type": "categorical",
            "values": ["adam", "sgd", "adamw", "rmsprop"]
        }
    }

    return search_spaces
```

### 2. Early Stopping Strategies

```python
def implement_early_stopping():
    """Implement effective early stopping strategies."""

    early_stopping_config = {
        "patience": 10,
        "min_delta": 0.001,
        "monitor": "validation_loss",
        "mode": "min"
    }

    def objective_with_early_stopping(trial: optuna.Trial) -> float:
        """Objective function with early stopping."""

        config = generate_config_from_trial(trial)

        with run.Experiment(f"early_stopping_{trial.number}") as exp:
            exp.add(config, name="early_stopping_trial")

            # Run with early stopping
            results = exp.run(
                early_stopping=early_stopping_config,
                max_epochs=100
            )

            return results["best_validation_accuracy"]

    return objective_with_early_stopping
```

### 3. Result Analysis and Visualization

```python
def analyze_optimization_results(study: optuna.Study):
    """Analyze and visualize optimization results."""

    # Plot optimization history
    optuna.visualization.plot_optimization_history(study)

    # Plot parameter importance
    optuna.visualization.plot_param_importances(study)

    # Plot parameter relationships
    optuna.visualization.plot_parallel_coordinate(study)

    # Get detailed statistics
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")

    # Analyze convergence
    convergence_analysis = {
        "trials_to_best": study.best_trial.number,
        "improvement_rate": calculate_improvement_rate(study),
        "parameter_sensitivity": analyze_parameter_sensitivity(study)
    }

    return convergence_analysis
```

## Performance Optimization
Run trials efficiently with parallelism, caching, and pruning.

### 1. Parallel Trial Execution

```python
def parallel_optimization():
    """Execute optimization trials in parallel."""

    # Configure parallel execution
    parallel_config = {
        "n_jobs": 4,
        "timeout": 3600,
        "catch": (Exception,)
    }

    study = optuna.create_study(direction="maximize")

    # Run parallel optimization
    study.optimize(
        objective,
        n_trials=100,
        n_jobs=4,
        timeout=3600
    )

    return study
```

### 2. Caching and Pruning

```python
def cached_optimization():
    """Use caching and pruning for efficient optimization."""

    # Create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )

    # Run with caching
    study.optimize(
        objective,
        n_trials=100,
        callbacks=[optuna.callbacks.CacheCallback()]
    )

    return study
```

## Monitoring and Debugging
Track progress and diagnose failed trials.

### 1. Optimization Progress Monitoring

```python
def monitor_optimization_progress(study: optuna.Study):
    """Monitor optimization progress in real-time."""

    def progress_callback(study: optuna.Study, trial: optuna.FrozenTrial):
        """Callback to monitor progress."""
        print(f"Trial {trial.number}: {trial.value}")
        print(f"Best so far: {study.best_value}")
        print(f"Parameters: {trial.params}")

    # Add callback to study
    study.optimize(
        objective,
        n_trials=100,
        callbacks=[progress_callback]
    )
```

### 2. Debugging Failed Trials

```python
def debug_failed_trials(study: optuna.Study):
    """Analyze and debug failed optimization trials."""

    failed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]

    print(f"Number of failed trials: {len(failed_trials)}")

    for trial in failed_trials:
        print(f"Trial {trial.number} failed:")
        print(f"  Parameters: {trial.params}")
        print(f"  Error: {trial.user_attrs.get('error', 'Unknown error')}")

    # Analyze failure patterns
    failure_patterns = analyze_failure_patterns(failed_trials)

    return failure_patterns
```

## Integration Examples
Tie optimizers to popular tracking tools.

### 1. Integration with MLflow

```python
def mlflow_integration():
    """Integrate hyperparameter optimization with MLflow tracking."""

    import mlflow

    def mlflow_objective(trial: optuna.Trial) -> float:
        """Objective function with MLflow tracking."""

        config = generate_config_from_trial(trial)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(trial.params)

            # Execute experiment
            with run.Experiment(f"mlflow_{trial.number}") as exp:
                exp.add(config, name="mlflow_trial")
                results = exp.run()

            # Log metrics
            mlflow.log_metrics(results)

            return results["validation_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(mlflow_objective, n_trials=100)

    return study
```

### 2. Integration with Weights & Biases

```python
def wandb_integration():
    """Integrate hyperparameter optimization with Weights & Biases."""

    import wandb

    def wandb_objective(trial: optuna.Trial) -> float:
        """Objective function with W&B tracking."""

        config = generate_config_from_trial(trial)

        # Initialize W&B run
        wandb.init(
            project="nemo-run-optimization",
            config=trial.params,
            name=f"trial_{trial.number}"
        )

        # Execute experiment
        with run.Experiment(f"wandb_{trial.number}") as exp:
            exp.add(config, name="wandb_trial")
            results = exp.run()

        # Log results
        wandb.log(results)
        wandb.finish()

        return results["validation_accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(wandb_objective, n_trials=100)

    return study
```

## Summary
Key benefits and why these approaches matter.

NeMo Run provides powerful tools for hyperparameter optimization that can significantly accelerate ML research and development. Key benefits include:

- **Advanced Optimization Algorithms**: Bayesian optimization, multi-objective optimization, and population-based methods
- **Resource Efficiency**: Early stopping, pruning, and multi-fidelity optimization
- **Scalability**: Distributed optimization across multiple nodes
- **Integration**: Seamless integration with popular ML tools and frameworks
- **Monitoring**: Comprehensive progress tracking and debugging capabilities

By leveraging these capabilities, researchers and ML engineers can automate the tedious process of hyperparameter tuning while achieving better results in less time.
