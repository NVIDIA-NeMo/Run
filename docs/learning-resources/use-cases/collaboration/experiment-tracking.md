---
description: "Track and share experiments across team members using NeMo Run"
categories: ["use-cases", "collaboration"]
tags: ["experiment-tracking", "metadata", "sharing", "collaboration"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "use-case"
modality: "text-only"
---

# Experiment Tracking

Track and share experiments across team members with centralized experiment management, metadata organization, and collaboration tools.

## Overview

Experiment tracking in NeMo Run provides comprehensive tools for managing experiment lifecycles, sharing results across teams, and maintaining organized metadata for effective collaboration and knowledge management.

## Key Features

### Centralized Experiment Management
- Single source of truth for all experiments
- Experiment metadata organization
- Version control for experiment configurations
- Experiment lifecycle tracking

### Metadata Organization
- Structured experiment metadata
- Custom metadata fields
- Tagging and categorization
- Search and filtering capabilities

### Result Sharing
- Automated result sharing
- Team notification systems
- Result visualization and reporting
- Collaboration tools integration

### Collaboration Tools
- Team experiment coordination
- Knowledge management
- Research documentation
- Training and onboarding

## Implementation
A practical, modular system you can adapt to your team.

### Experiment Tracking System

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

@dataclass
class ExperimentMetadata:
    """Structured experiment metadata for tracking."""

    experiment_id: str
    name: str
    description: str
    researcher: str
    team: str
    project: str
    status: str  # running, completed, failed, cancelled
    created_date: str
    updated_date: str
    tags: List[str]
    metrics: Dict[str, Any]
    configuration: Dict[str, Any]
    artifacts: List[str]
    notes: str
    collaborators: List[str]

    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "researcher": self.researcher,
            "team": self.team,
            "project": self.project,
            "status": self.status,
            "created_date": self.created_date,
            "updated_date": self.updated_date,
            "tags": self.tags,
            "metrics": self.metrics,
            "configuration": self.configuration,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "collaborators": self.collaborators
        }

class ExperimentTracker:
    """Centralized experiment tracking system."""

    def __init__(self, team_name: str):
        self.team_name = team_name
        self.experiments_repository = f"experiments/{team_name}"
        self.metadata_file = f"{self.experiments_repository}/metadata.json"
        self._initialize_repository()

    def _initialize_repository(self):
        """Initialize the experiment repository."""
        import os
        os.makedirs(self.experiments_repository, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            import json
            with open(self.metadata_file, 'w') as f:
                json.dump({"experiments": {}}, f)

    def track_experiment(self, experiment_name: str, description: str,
                        configuration: Dict[str, Any], tags: List[str] = None):
        """Track a new experiment."""

        import uuid
        import datetime
        experiment_id = str(uuid.uuid4())
        researcher = os.getenv('USER', 'unknown')

        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=experiment_name,
            description=description,
            researcher=researcher,
            team=self.team_name,
            project=os.getcwd().split('/')[-1],  # Use current directory as project
            status="running",
            created_date=datetime.datetime.now().isoformat(),
            updated_date=datetime.datetime.now().isoformat(),
            tags=tags or [],
            metrics={},
            configuration=configuration,
            artifacts=[],
            notes="",
            collaborators=[]
        )

        # Store metadata
        self._store_metadata(metadata)

        # Notify team
        self._notify_team_new_experiment(metadata)

        return metadata

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        """Update experiment metadata."""

        metadata = self._load_metadata()
        if experiment_id not in metadata["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Update fields
        experiment = metadata["experiments"][experiment_id]
        for key, value in updates.items():
            if hasattr(ExperimentMetadata, key):
                experiment[key] = value

        experiment["updated_date"] = datetime.datetime.now().isoformat()

        # Store updated metadata
        self._store_metadata_dict(metadata)

        return experiment

    def complete_experiment(self, experiment_id: str, metrics: Dict[str, Any],
                          artifacts: List[str] = None):
        """Mark experiment as completed with results."""

        updates = {
            "status": "completed",
            "metrics": metrics,
            "artifacts": artifacts or []
        }

        return self.update_experiment(experiment_id, updates)

    def share_experiment(self, experiment_id: str, collaborators: List[str]):
        """Share experiment with team members."""

        updates = {
            "collaborators": collaborators,
            "status": "shared"
        }

        experiment = self.update_experiment(experiment_id, updates)

        # Notify collaborators
        self._notify_collaborators(experiment, collaborators)

        return experiment

    def search_experiments(self, query: str = None, tags: List[str] = None,
                          researcher: str = None, status: str = None):
        """Search experiments by various criteria."""

        metadata = self._load_metadata()
        experiments = metadata["experiments"].values()

        filtered_experiments = []

        for experiment in experiments:
            # Apply filters
            if query and query.lower() not in experiment["name"].lower() and \
               query.lower() not in experiment["description"].lower():
                continue

            if tags and not any(tag in experiment["tags"] for tag in tags):
                continue

            if researcher and experiment["researcher"] != researcher:
                continue

            if status and experiment["status"] != status:
                continue

            filtered_experiments.append(experiment)

        return filtered_experiments

    def get_experiment_summary(self):
        """Get summary statistics for team experiments."""

        metadata = self._load_metadata()
        experiments = metadata["experiments"].values()

        summary = {
            "total_experiments": len(experiments),
            "completed_experiments": len([e for e in experiments if e["status"] == "completed"]),
            "running_experiments": len([e for e in experiments if e["status"] == "running"]),
            "failed_experiments": len([e for e in experiments if e["status"] == "failed"]),
            "researchers": list(set(e["researcher"] for e in experiments)),
            "projects": list(set(e["project"] for e in experiments)),
            "tags": list(set(tag for e in experiments for tag in e["tags"]))
        }

        return summary

    def _store_metadata(self, metadata: ExperimentMetadata):
        """Store experiment metadata."""

        metadata_dict = self._load_metadata()
        metadata_dict["experiments"][metadata.experiment_id] = metadata.to_dict()
        self._store_metadata_dict(metadata_dict)

    def _load_metadata(self):
        """Load experiment metadata."""

        import json
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _store_metadata_dict(self, metadata_dict: Dict[str, Any]):
        """Store metadata dictionary."""

        import json
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def _notify_team_new_experiment(self, metadata: ExperimentMetadata):
        """Notify team about new experiment."""

        notification = {
            "type": "new_experiment",
            "experiment_id": metadata.experiment_id,
            "name": metadata.name,
            "researcher": metadata.researcher,
            "description": metadata.description,
            "timestamp": metadata.created_date
        }

        # Note: Notification system not implemented in NeMo Run
        # This would integrate with your team's notification system
        print(f"Notification sent to team {self.team_name}: {notification}")

    def _notify_collaborators(self, experiment: Dict[str, Any], collaborators: List[str]):
        """Notify collaborators about shared experiment."""

        notification = {
            "type": "shared_experiment",
            "experiment_id": experiment["experiment_id"],
            "name": experiment["name"],
            "researcher": experiment["researcher"],
            "collaborators": collaborators,
            "timestamp": experiment["updated_date"]
        }

        for collaborator in collaborators:
            # Note: Notification system not implemented in NeMo Run
            # This would integrate with your team's notification system
            print(f"Notification sent to {collaborator}: {notification}")

# Initialize experiment tracker
experiment_tracker = ExperimentTracker("ml_team")
```

### Automated Experiment Tracking

```python
class AutomatedTracker:
    """Automated experiment tracking with NeMo Run integration."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def track_experiment_run(self, experiment_name: str, description: str,
                           configuration: Dict[str, Any], tags: List[str] = None):
        """Automatically track experiment during execution."""

        # Start tracking
        metadata = self.tracker.track_experiment(experiment_name, description, configuration, tags)

        # Execute experiment with tracking
        with run.Experiment(experiment_name) as exp:

            # Add configuration
            exp.add(configuration, name="main_experiment")

            # Add tracking callbacks
            exp.add_callback(self._track_progress_callback(metadata.experiment_id))
            exp.add_callback(self._track_metrics_callback(metadata.experiment_id))

            # Execute experiment
            results = exp.run()

            # Complete tracking
            self.tracker.complete_experiment(
                metadata.experiment_id,
                results.metrics,
                results.artifacts
            )

            return results

    def _track_progress_callback(self, experiment_id: str):
        """Callback to track experiment progress."""

        def progress_callback(progress_data):
            # Update experiment status
            self.tracker.update_experiment(experiment_id, {
                "status": "running",
                "progress": progress_data
            })

        return progress_callback

    def _track_metrics_callback(self, experiment_id: str):
        """Callback to track experiment metrics."""

        def metrics_callback(metrics_data):
            # Update experiment metrics
            self.tracker.update_experiment(experiment_id, {
                "metrics": metrics_data
            })

        return metrics_callback

# Initialize automated tracker
automated_tracker = AutomatedTracker(experiment_tracker)
```

### Team Collaboration Features

```python
class TeamCollaboration:
    """Team collaboration features for experiment tracking."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def share_experiment_with_team(self, experiment_id: str, team_members: List[str]):
        """Share experiment with team members."""

        return self.tracker.share_experiment(experiment_id, team_members)

    def create_team_dashboard(self):
        """Create team experiment dashboard."""

        summary = self.tracker.get_experiment_summary()
        recent_experiments = self.tracker.search_experiments(status="completed")[:10]

        dashboard = {
            "summary": summary,
            "recent_experiments": recent_experiments,
            "team_performance": self._calculate_team_performance(),
            "project_progress": self._calculate_project_progress()
        }

        return dashboard

    def generate_team_report(self, time_period: str = "month"):
        """Generate team experiment report."""

        experiments = self.tracker.search_experiments(status="completed")

        report = {
            "period": time_period,
            "total_experiments": len(experiments),
            "successful_experiments": len([e for e in experiments if e["status"] == "completed"]),
            "average_metrics": self._calculate_average_metrics(experiments),
            "top_performers": self._identify_top_performers(experiments),
            "project_distribution": self._calculate_project_distribution(experiments),
            "collaboration_stats": self._calculate_collaboration_stats(experiments)
        }

        return report

    def _calculate_team_performance(self):
        """Calculate team performance metrics."""

        experiments = self.tracker.search_experiments(status="completed")

        performance = {
            "total_experiments": len(experiments),
            "success_rate": len(experiments) / max(len(experiments), 1),
            "average_experiment_time": self._calculate_average_time(experiments),
            "top_metrics": self._get_top_metrics(experiments)
        }

        return performance

    def _calculate_project_progress(self):
        """Calculate progress for each project."""

        projects = {}
        experiments = self.tracker.search_experiments()

        for experiment in experiments:
            project = experiment["project"]
            if project not in projects:
                projects[project] = {
                    "total": 0,
                    "completed": 0,
                    "running": 0,
                    "failed": 0
                }

            projects[project]["total"] += 1
            projects[project][experiment["status"]] += 1

        return projects

    def _calculate_average_metrics(self, experiments):
        """Calculate average metrics across experiments."""

        if not experiments:
            return {}

        all_metrics = {}
        for experiment in experiments:
            for metric, value in experiment["metrics"].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        average_metrics = {}
        for metric, values in all_metrics.items():
            average_metrics[metric] = sum(values) / len(values)

        return average_metrics

    def _identify_top_performers(self, experiments):
        """Identify top performing experiments."""

        # Sort by performance metrics
        sorted_experiments = sorted(
            experiments,
            key=lambda e: e.get("metrics", {}).get("accuracy", 0),
            reverse=True
        )

        return sorted_experiments[:5]

    def _calculate_project_distribution(self, experiments):
        """Calculate distribution of experiments across projects."""

        distribution = {}
        for experiment in experiments:
            project = experiment["project"]
            distribution[project] = distribution.get(project, 0) + 1

        return distribution

    def _calculate_collaboration_stats(self, experiments):
        """Calculate collaboration statistics."""

        collaboration_stats = {
            "shared_experiments": len([e for e in experiments if e["collaborators"]]),
            "average_collaborators": 0,
            "most_collaborative_researcher": None
        }

        if experiments:
            total_collaborators = sum(len(e["collaborators"]) for e in experiments)
            collaboration_stats["average_collaborators"] = total_collaborators / len(experiments)

            # Find most collaborative researcher
            researcher_collaborations = {}
            for experiment in experiments:
                researcher = experiment["researcher"]
                if researcher not in researcher_collaborations:
                    researcher_collaborations[researcher] = 0
                researcher_collaborations[researcher] += len(experiment["collaborators"])

            if researcher_collaborations:
                most_collaborative = max(researcher_collaborations.items(), key=lambda x: x[1])
                collaboration_stats["most_collaborative_researcher"] = most_collaborative[0]

        return collaboration_stats

    def _calculate_average_time(self, experiments):
        """Calculate average experiment time."""

        if not experiments:
            return 0

        total_time = 0
        count = 0

        for experiment in experiments:
            created = datetime.fromisoformat(experiment["created_date"])
            updated = datetime.fromisoformat(experiment["updated_date"])
            duration = (updated - created).total_seconds()
            total_time += duration
            count += 1

        return total_time / count if count > 0 else 0

    def _get_top_metrics(self, experiments):
        """Get top performing metrics."""

        if not experiments:
            return {}

        # Find best performing experiment for each metric
        top_metrics = {}
        for experiment in experiments:
            for metric, value in experiment["metrics"].items():
                if metric not in top_metrics or value > top_metrics[metric]["value"]:
                    top_metrics[metric] = {
                        "value": value,
                        "experiment": experiment["name"],
                        "researcher": experiment["researcher"]
                    }

        return top_metrics

# Initialize team collaboration
team_collaboration = TeamCollaboration(experiment_tracker)
```

## Use Cases
Patterns for research and product teams.

### Research Team Experiment Tracking

**Scenario**: Academic research team tracking experiments for publication

**Implementation**:
```python
# Research team experiment tracking
research_tracker = ExperimentTracker("neural_network_research")

# Track a research experiment
experiment_config = {
    "model": "transformer",
    "dataset": "wmt14",
    "optimizer": "adam",
    "learning_rate": 0.001
}

metadata = research_tracker.track_experiment(
    "transformer_wmt14_optimization",
    "Optimizing transformer model on WMT14 dataset",
    experiment_config,
    tags=["transformer", "machine_translation", "optimization"]
)

# Execute with automated tracking
automated_tracker = AutomatedTracker(research_tracker)
results = automated_tracker.track_experiment_run(
    metadata.name,
    metadata.description,
    metadata.configuration,
    metadata.tags
)

# Share with research team
research_tracker.share_experiment(
    metadata.experiment_id,
    ["alice", "bob", "charlie"]
)
```

### Industry Team Collaboration

**Scenario**: Industry team tracking experiments for product development

**Implementation**:
```python
# Industry team experiment tracking
product_tracker = ExperimentTracker("recommendation_system")

# Track product experiment
product_config = {
    "model_type": "collaborative_filtering",
    "embedding_dim": 128,
    "regularization": 0.01,
    "evaluation_metrics": ["precision", "recall", "ndcg"]
}

metadata = product_tracker.track_experiment(
    "recommendation_optimization_v2",
    "Optimizing recommendation system for better user engagement",
    product_config,
    tags=["recommendation", "optimization", "production"]
)

# Execute with tracking
results = automated_tracker.track_experiment_run(
    metadata.name,
    metadata.description,
    metadata.configuration,
    metadata.tags
)

# Generate team report
team_collab = TeamCollaboration(product_tracker)
dashboard = team_collab.create_team_dashboard()
report = team_collab.generate_team_report("month")
```

## Best Practices
Recommendations to keep experiments organized and shareable.

### 1. Metadata Organization
- Use consistent naming conventions
- Implement comprehensive tagging
- Maintain detailed descriptions
- Track all relevant metrics

### 2. Collaboration
- Share experiments proactively
- Document findings and insights
- Encourage team feedback
- Maintain knowledge base

### 3. Quality Assurance
- Validate experiment results
- Review and approve experiments
- Maintain experiment history
- Ensure reproducibility

### 4. Reporting
- Generate regular team reports
- Track performance metrics
- Identify improvement opportunities
- Share best practices

## Success Metrics
Measure adoption, quality, and collaboration effectiveness.

### Experiment Management
- **Experiment completion rate**: Percentage of experiments completed successfully
- **Average experiment time**: Time from start to completion
- **Experiment quality**: Performance metrics and reproducibility
- **Knowledge retention**: Long-term experiment knowledge preservation

### Collaboration
- **Team participation**: Number of team members contributing experiments
- **Knowledge sharing**: Effectiveness of experiment sharing
- **Collaboration efficiency**: Time to complete collaborative experiments
- **Team satisfaction**: Survey-based satisfaction scores

### Quality Metrics
- **Experiment reproducibility**: Percentage of reproducible experiments
- **Documentation quality**: Completeness of experiment documentation
- **Result consistency**: Variance in experiment performance
- **Error reduction**: Decrease in experiment failures

## Next Steps
Where to go next for deeper integrations and patterns.

- Explore **[Configuration Guide](../../../guides/configuration.md)** for collaborative development patterns
- Check **[Package Code for Deployment](../../../guides/packaging.md)** for reusable component patterns
- Review **[Guides](../../../guides/index.md)** for optimization strategies
- Consult **[References](../../../references/index.md)** for detailed API documentation
