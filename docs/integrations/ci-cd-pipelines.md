---
description: "Integration guides for connecting NeMo Run with CI/CD pipelines like GitHub Actions, GitLab CI, Jenkins, and other automation tools."
categories: ["integrations-apis"]
tags: ["integrations", "ci-cd", "github-actions", "gitlab-ci", "jenkins", "automation"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "text-only"
---

(ci-cd-pipelines)=

# CI/CD Pipelines Integration

This guide covers integrating NeMo Run with Continuous Integration/Continuous Deployment (CI/CD) pipelines to automate ML experiment execution, testing, and deployment.

## Supported CI/CD Platforms

NeMo Run integrates with popular CI/CD platforms:

- **GitHub Actions** - GitHub's native CI/CD platform
- **GitLab CI/CD** - GitLab's integrated CI/CD solution
- **Jenkins** - Open-source automation server
- **Azure DevOps** - Microsoft's DevOps platform
- **CircleCI** - Cloud-based CI/CD platform
- **Travis CI** - Continuous integration service

## GitHub Actions Integration

### Basic GitHub Actions Workflow

```yaml
# .github/workflows/nemo-run-tests.yml
name: NeMo Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install NeMo Run
      run: |
        pip install git+https://github.com/NVIDIA-NeMo/Run.git

    - name: Run basic tests
      run: |
        python -c "
        import nemo_run as run
        print('✅ NeMo Run imported successfully')

        # Test basic configuration
        def test_function(x):
            return x * 2

        config = run.Config(test_function, x=42)
        result = config.build()
        print(f'✅ Configuration test passed: {result}')
        "

    - name: Run experiment tests
      run: |
        python tests/test_experiments.py
```

### Advanced GitHub Actions with GPU Support

```yaml
# .github/workflows/nemo-run-gpu-tests.yml
name: NeMo Run GPU Tests

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  gpu-test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        cuda-version: [11.8, 12.1]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install CUDA
      uses: Jimver/cuda-toolkit@v0.2.7
      with:
        cuda: ${{ matrix.cuda-version }}

    - name: Install NeMo Run
      run: |
        pip install git+https://github.com/NVIDIA-NeMo/Run.git
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${{ matrix.cuda-version }}

    - name: Run GPU tests
      run: |
        python -c "
        import nemo_run as run

        def gpu_test():
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {device}')
            return torch.cuda.is_available()

        # Test with Docker executor
        executor = run.DockerExecutor(
            container_image='nvidia/pytorch:24.05-py3',
            num_gpus=1
        )

        config = run.Partial(gpu_test)

        with run.Experiment('gpu_test') as experiment:
            experiment.add(config, executor=executor, name='gpu_test')
            experiment.run()
        "
```

### Automated Model Training Pipeline

```yaml
# .github/workflows/automated-training.yml
name: Automated Model Training

on:
  push:
    branches: [ main ]
    paths: [ 'models/**', 'configs/**' ]
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install git+https://github.com/NVIDIA-NeMo/Run.git
        pip install torch torchvision

    - name: Run training experiment
      run: |
        python -c "
        import nemo_run as run

        def train_model(model_config, epochs=10):
            print(f'Training model with {epochs} epochs')
            # Training logic here
            return {'accuracy': 0.95, 'loss': 0.05}

        # Create model configuration
        model_config = run.Config(train_model, epochs=20)

        # Use Docker executor for consistent environment
        executor = run.DockerExecutor(
            container_image='nvidia/pytorch:24.05-py3',
            num_gpus=1
        )

        # Run experiment
        with run.Experiment('automated_training') as experiment:
            experiment.add(model_config, executor=executor, name='training_run')
            experiment.run()
        "

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: training-results
        path: |
          ~/.nemo_run/experiments/automated_training/
```

## GitLab CI Integration

### Basic GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - train

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

cache:
  paths:
    - .pip-cache

test:
  stage: test
  image: python:3.10
  script:
    - pip install git+https://github.com/NVIDIA-NeMo/Run.git
    - python -c "
      import nemo_run as run
      print('✅ NeMo Run tests passed')
    "
  only:
    - merge_requests
    - main

train:
  stage: train
  image: nvidia/cuda:11.8-devel-ubuntu20.04
  services:
    - docker:dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  script:
    - apt-get update && apt-get install -y python3-pip
    - pip install git+https://github.com/NVIDIA-NeMo/Run.git
    - python3 -c "
      import nemo_run as run

      def train_function():
          print('Training model in GitLab CI')
          return {'status': 'completed'}

      config = run.Partial(train_function)

      with run.Experiment('gitlab_training') as experiment:
          experiment.add(config, name='training')
          experiment.run()
      "
  only:
    - main
```

## Jenkins Integration

### Jenkins Pipeline Script

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        PYTHON_VERSION = '3.10'
        NEMO_RUN_HOME = "${WORKSPACE}/.nemo_run"
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip install git+https://github.com/NVIDIA-NeMo/Run.git
                    pip install torch torchvision
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    python3 -c "
                    import nemo_run as run
                    print('✅ NeMo Run imported successfully')

                    def test_function():
                        return 'test passed'

                    config = run.Partial(test_function)
                    result = config.build()()
                    print(f'Result: {result}')
                    "
                '''
            }
        }

        stage('Train') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    python3 -c "
                    import nemo_run as run

                    def train_model():
                        print('Training model in Jenkins')
                        return {'accuracy': 0.92}

                    config = run.Partial(train_model)

                    with run.Experiment('jenkins_training') as experiment:
                        experiment.add(config, name='training')
                        experiment.run()
                    "
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '.nemo_run/experiments/**/*', fingerprint: true
        }
    }
}
```

## Azure DevOps Integration

### Azure DevOps Pipeline

```yaml
# azure-pipelines.yml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  python.version: '3.10'

stages:
- stage: Test
  displayName: 'Test Stage'
  jobs:
  - job: Test
    displayName: 'Run Tests'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(python.version)'
        addToPath: true

    - script: |
        python -m pip install --upgrade pip
        pip install git+https://github.com/NVIDIA-NeMo/Run.git
      displayName: 'Install NeMo Run'

    - script: |
        python -c "
        import nemo_run as run
        print('✅ NeMo Run tests completed')
        "
      displayName: 'Run Tests'

- stage: Train
  displayName: 'Training Stage'
  dependsOn: Test
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - job: Train
    displayName: 'Train Model'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(python.version)'
        addToPath: true

    - script: |
        pip install git+https://github.com/NVIDIA-NeMo/Run.git
        pip install torch torchvision
      displayName: 'Install Dependencies'

    - script: |
        python -c "
        import nemo_run as run

        def train_model():
            print('Training model in Azure DevOps')
            return {'status': 'success'}

        config = run.Partial(train_model)

        with run.Experiment('azure_training') as experiment:
            experiment.add(config, name='training')
            experiment.run()
        "
      displayName: 'Train Model'

    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: '.nemo_run/experiments'
        artifactName: 'training-results'
```

## CircleCI Integration

### CircleCI Configuration

```yaml
# .circleci/config.yml
version: 2.1

orbs:
  python: circleci/python@2.1

jobs:
  test:
    docker:
      - image: cimg/python:3.10
    steps:
      - checkout
      - python/install-packages:
          pkg-manager: pip
          app-dir: ./
      - run:
          name: Install NeMo Run
          command: |
            pip install git+https://github.com/NVIDIA-NeMo/Run.git
      - run:
          name: Run Tests
          command: |
            python -c "
            import nemo_run as run
            print('✅ NeMo Run tests passed')
            "

  train:
    docker:
      - image: cimg/python:3.10
    steps:
      - checkout
      - python/install-packages:
          pkg-manager: pip
          app-dir: ./
      - run:
          name: Install Dependencies
          command: |
            pip install git+https://github.com/NVIDIA-NeMo/Run.git
            pip install torch torchvision
      - run:
          name: Train Model
          command: |
            python -c "
            import nemo_run as run

            def train_model():
                print('Training model in CircleCI')
                return {'accuracy': 0.94}

            config = run.Partial(train_model)

            with run.Experiment('circleci_training') as experiment:
                experiment.add(config, name='training')
                experiment.run()
            "
      - store_artifacts:
          path: .nemo_run/experiments
          destination: training-results

workflows:
  version: 2
  test-and-train:
    jobs:
      - test
      - train:
          requires:
            - test
          filters:
            branches:
              only: main
```

## Best Practices

### Environment Management

```python
# ci_utils.py
import os
import nemo_run as run

def get_ci_executor():
    """Get appropriate executor based on CI environment."""
    ci_platform = os.getenv('CI_PLATFORM', 'local')

    if ci_platform == 'github':
        return run.DockerExecutor(
            container_image='nvidia/pytorch:24.05-py3',
            num_gpus=1
        )
    elif ci_platform == 'gitlab':
        return run.LocalExecutor()
    else:
        return run.LocalExecutor()

def run_ci_experiment(experiment_name, config):
    """Run experiment with CI-appropriate settings."""
    executor = get_ci_executor()

    with run.Experiment(experiment_name) as experiment:
        experiment.add(config, executor=executor, name='ci_run')
        experiment.run()
```

### Automated Testing

```python
# test_automation.py
import nemo_run as run

def test_configuration():
    """Test NeMo Run configuration system."""
    def test_function(x, y):
        return x + y

    config = run.Config(test_function, x=10, y=20)
    result = config.build()()
    assert result == 30
    print("✅ Configuration test passed")

def test_experiment():
    """Test experiment execution."""
    def dummy_training():
        return {'status': 'completed'}

    config = run.Partial(dummy_training)

    with run.Experiment('test_experiment') as experiment:
        experiment.add(config, name='test_run')
        experiment.run()

    print("✅ Experiment test passed")

if __name__ == "__main__":
    test_configuration()
    test_experiment()
```

### Error Handling

```python
# ci_error_handling.py
import sys
import nemo_run as run

def safe_ci_execution():
    """Execute NeMo Run with proper error handling."""
    try:
        def training_function():
            # Simulate training
            return {'accuracy': 0.95}

        config = run.Partial(training_function)

        with run.Experiment('ci_training') as experiment:
            experiment.add(config, name='training')
            experiment.run()

        print("✅ CI execution completed successfully")
        return True

    except Exception as e:
        print(f"❌ CI execution failed: {e}")
        sys.exit(1)
```

## Integration Checklist

When integrating NeMo Run with CI/CD pipelines:

- [ ] **Environment Setup**: Ensure Python and dependencies are properly installed
- [ ] **GPU Support**: Configure GPU access if needed for training
- [ ] **Artifact Management**: Set up proper artifact storage and retrieval
- [ ] **Error Handling**: Implement robust error handling and reporting
- [ ] **Security**: Secure sensitive data and credentials
- [ ] **Monitoring**: Set up proper logging and monitoring
- [ ] **Testing**: Include comprehensive testing in the pipeline

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure NeMo Run is properly installed
2. **GPU Access**: Verify GPU drivers and CUDA installation
3. **Permission Issues**: Check file permissions and access rights
4. **Memory Issues**: Monitor resource usage and adjust accordingly

### Debug Commands

```bash
# Check NeMo Run installation
python -c "import nemo_run as run; print(run.__version__)"

# Test basic functionality
python -c "
import nemo_run as run
def test(): return 'success'
config = run.Partial(test)
result = config.build()()
print(result)
"

# Check experiment status
python -c "
import nemo_run as run
experiment = run.Experiment.from_title('test_experiment')
experiment.status()
"
```

## Next Steps

- Review [Best Practices](../best-practices/index) for production CI/CD workflows
- Explore [Cloud Platform Integration](cloud-platforms) for cloud-based execution
- Check [Monitoring Tools Integration](monitoring-tools) for experiment tracking
